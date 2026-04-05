import CoreML
import Foundation

public struct GenerationConfig: Sendable {
    public var maxNewTokens: Int
    public var temperature: Float
    public var topK: Int

    public init(maxNewTokens: Int = 128, temperature: Float = 0, topK: Int = 40) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topK = topK
    }
}

public enum LFM2PipelineError: Error {
    case contextLimitExceeded(limit: Int)
    case unsupportedImages
    case invalidModelOutput(String)
    case emptyPrompt
}

public final class LFM2Pipeline {
    public let bundle: LFM2ModelBundle

    private var convCaches: [MLMultiArray] = []
    private var kCaches: [MLMultiArray] = []
    private var vCaches: [MLMultiArray] = []
    private var currentPosition = 0
    private var lastLogits: [Float] = []

    public init(bundle: LFM2ModelBundle) throws {
        self.bundle = bundle
        try reset()
    }

    public func reset() throws {
        convCaches = try bundle.metadata.chunks.map { chunk in
            try TensorHelpers.makeFloat16Array(
                shape: [chunk.convLayerIndices.count, bundle.metadata.hiddenSize, bundle.metadata.convKernelSize]
            )
        }
        kCaches = try bundle.metadata.chunks.map { chunk in
            try TensorHelpers.makeFloat16Array(
                shape: [chunk.attentionLayerIndices.count, bundle.metadata.numKeyValueHeads, bundle.metadata.contextLength, bundle.metadata.headDim]
            )
        }
        vCaches = try bundle.metadata.chunks.map { chunk in
            try TensorHelpers.makeFloat16Array(
                shape: [chunk.attentionLayerIndices.count, bundle.metadata.numKeyValueHeads, bundle.metadata.contextLength, bundle.metadata.headDim]
            )
        }
        currentPosition = 0
        lastLogits = []
    }

    public func generate(messages: [ChatMessage], config: GenerationConfig = .init()) async throws -> String {
        if messages.isEmpty {
            throw LFM2PipelineError.emptyPrompt
        }
        if messages.contains(where: { $0.content.contains(.imagePlaceholder) }) {
            throw LFM2PipelineError.unsupportedImages
        }

        try reset()
        let rendered = try ChatFormatter.render(messages: messages, addGenerationPrompt: true)
        var promptTokens = bundle.tokenizer.encode(text: rendered, addSpecialTokens: false)
        if promptTokens.count >= bundle.metadata.contextLength {
            promptTokens = Array(promptTokens.suffix(bundle.metadata.contextLength - 1))
        }
        guard !promptTokens.isEmpty else {
            throw LFM2PipelineError.emptyPrompt
        }

        try prefill(tokens: promptTokens)

        var generatedTokenIDs: [Int] = []
        var nextToken = try nextTokenFromCurrentState(config: config)

        while generatedTokenIDs.count < config.maxNewTokens {
            if nextToken == bundle.metadata.eosTokenId {
                break
            }
            generatedTokenIDs.append(nextToken)
            if currentPosition >= bundle.metadata.contextLength {
                break
            }
            try decode(tokenID: nextToken)
            nextToken = try nextTokenFromCurrentState(config: config)
        }

        return bundle.tokenizer.decode(tokens: generatedTokenIDs, skipSpecialTokens: true)
    }

    private func prefill(tokens: [Int]) throws {
        var offset = 0
        while offset < tokens.count {
            let remaining = tokens.count - offset
            if remaining >= bundle.metadata.prefillLength {
                let block = Array(tokens[offset..<(offset + bundle.metadata.prefillLength)])
                try runPrefillBlock(block, startPosition: offset)
                offset += bundle.metadata.prefillLength
            } else {
                for token in tokens[offset...] {
                    try decode(tokenID: token)
                }
                offset = tokens.count
            }
        }
    }

    private func runPrefillBlock(_ tokenIDs: [Int], startPosition: Int) throws {
        let embeddingsOutput = try predict(
            model: bundle.embeddingsModel,
            inputs: ["input_ids": try TensorHelpers.makeInt32Array(shape: [1, tokenIDs.count], values: tokenIDs)]
        )
        let hiddenStates = try multiArray(embeddingsOutput, name: "embeddings")
        let positionIDs = try TensorHelpers.makeInt32Array(
            shape: [1, tokenIDs.count],
            values: Array(startPosition..<(startPosition + tokenIDs.count))
        )
        let causalMask = try TensorHelpers.makeFloat16Array(
            shape: [tokenIDs.count, bundle.metadata.contextLength],
            values: makeCausalMask(startPosition: startPosition, count: tokenIDs.count)
        )
        let writeMask = try TensorHelpers.makeFloat16Array(
            shape: [tokenIDs.count, bundle.metadata.contextLength],
            values: makeWriteMask(startPosition: startPosition, count: tokenIDs.count)
        )

        var currentHidden = hiddenStates
        for chunkIndex in bundle.metadata.chunks.indices {
            let outputs = try predict(
                model: bundle.prefillModels[chunkIndex],
                inputs: [
                    "hidden_states": currentHidden,
                    "position_ids": positionIDs,
                    "causal_mask": causalMask,
                    "write_mask": writeMask,
                    "conv_cache": convCaches[chunkIndex],
                    "k_cache": kCaches[chunkIndex],
                    "v_cache": vCaches[chunkIndex]
                ]
            )
            currentHidden = try multiArray(outputs, name: "hidden_states_out")
            convCaches[chunkIndex] = try multiArray(outputs, name: "conv_cache_out")
            kCaches[chunkIndex] = try multiArray(outputs, name: "k_cache_out")
            vCaches[chunkIndex] = try multiArray(outputs, name: "v_cache_out")
        }
        lastLogits = try logits(from: currentHidden)
        currentPosition = startPosition + tokenIDs.count
    }

    private func decode(tokenID: Int) throws {
        guard currentPosition < bundle.metadata.contextLength else {
            throw LFM2PipelineError.contextLimitExceeded(limit: bundle.metadata.contextLength)
        }
        let embeddingsOutput = try predict(
            model: bundle.embeddingsModel,
            inputs: ["input_ids": try TensorHelpers.makeInt32Array(shape: [1, 1], values: [tokenID])]
        )
        let hiddenStates = try multiArray(embeddingsOutput, name: "embeddings")
        let positionIDs = try TensorHelpers.makeInt32Array(shape: [1, 1], values: [currentPosition])
        let causalMask = try TensorHelpers.makeFloat16Array(
            shape: [1, bundle.metadata.contextLength],
            values: makeCausalMask(startPosition: currentPosition, count: 1)
        )
        let writeMask = try TensorHelpers.makeFloat16Array(
            shape: [1, bundle.metadata.contextLength],
            values: makeWriteMask(startPosition: currentPosition, count: 1)
        )

        var currentHidden = hiddenStates
        for chunkIndex in bundle.metadata.chunks.indices {
            let outputs = try predict(
                model: bundle.decodeModels[chunkIndex],
                inputs: [
                    "hidden_states": currentHidden,
                    "position_ids": positionIDs,
                    "causal_mask": causalMask,
                    "write_mask": writeMask,
                    "conv_cache": convCaches[chunkIndex],
                    "k_cache": kCaches[chunkIndex],
                    "v_cache": vCaches[chunkIndex]
                ]
            )
            currentHidden = try multiArray(outputs, name: "hidden_states_out")
            convCaches[chunkIndex] = try multiArray(outputs, name: "conv_cache_out")
            kCaches[chunkIndex] = try multiArray(outputs, name: "k_cache_out")
            vCaches[chunkIndex] = try multiArray(outputs, name: "v_cache_out")
        }
        lastLogits = try logits(from: currentHidden)
        currentPosition += 1
    }

    private func nextTokenFromCurrentState(config: GenerationConfig = .init()) throws -> Int {
        guard !lastLogits.isEmpty else {
            throw LFM2PipelineError.invalidModelOutput("logits")
        }
        return sampleToken(from: lastLogits, temperature: config.temperature, topK: config.topK)
    }

    private func predict(model: MLModel, inputs: [String: Any]) throws -> MLFeatureProvider {
        try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: inputs))
    }

    private func logits(from hiddenStates: MLMultiArray) throws -> [Float] {
        let logitsFeatures = try predict(model: bundle.lmHeadModel, inputs: ["hidden_states": hiddenStates])
        let logits = try multiArray(logitsFeatures, name: "logits")
        return TensorHelpers.lastTokenLogits(from: logits, vocabSize: bundle.metadata.vocabSize)
    }

    private func multiArray(_ outputs: MLFeatureProvider, name: String) throws -> MLMultiArray {
        guard let value = outputs.featureValue(for: name)?.multiArrayValue else {
            throw LFM2PipelineError.invalidModelOutput(name)
        }
        return value
    }

    private func makeCausalMask(startPosition: Int, count: Int) -> [Float] {
        var values = Array(repeating: Float(-10_000), count: count * bundle.metadata.contextLength)
        for row in 0..<count {
            let limit = min(startPosition + row, bundle.metadata.contextLength - 1)
            for column in 0...limit {
                values[(row * bundle.metadata.contextLength) + column] = 0
            }
        }
        return values
    }

    private func makeWriteMask(startPosition: Int, count: Int) -> [Float] {
        var values = Array(repeating: Float(0), count: count * bundle.metadata.contextLength)
        for row in 0..<count {
            let column = startPosition + row
            values[(row * bundle.metadata.contextLength) + column] = 1
        }
        return values
    }

    private func sampleToken(from logits: [Float], temperature: Float, topK: Int) -> Int {
        if temperature <= 0 {
            return logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? bundle.metadata.eosTokenId
        }

        let ranked = logits.enumerated().sorted(by: { $0.element > $1.element })
        let candidates = Array(ranked.prefix(max(topK, 1)))
        let maxLogit = candidates.first?.element ?? 0
        let scaled = candidates.map { exp(($0.element - maxLogit) / temperature) }
        let sum = scaled.reduce(0, +)
        let normalized = scaled.map { $0 / sum }
        let threshold = Float.random(in: 0..<1)
        var running: Float = 0
        for (index, candidate) in candidates.enumerated() {
            running += normalized[index]
            if threshold <= running {
                return candidate.offset
            }
        }
        return candidates.last?.offset ?? bundle.metadata.eosTokenId
    }
}
