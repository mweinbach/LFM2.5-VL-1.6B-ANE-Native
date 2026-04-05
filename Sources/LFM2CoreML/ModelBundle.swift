import CoreML
import Foundation
import Tokenizers

public struct LFM2ModelBundle {
    public let rootURL: URL
    public let tokenizerFolderURL: URL
    public let coreMLFolderURL: URL
    public let metadata: LFM2Metadata
    public let tokenizer: any Tokenizer
    public let embeddingsModel: MLModel
    public let lmHeadModel: MLModel
    public let visionPatchEmbeddingModel: MLModel
    public let visionEncoderModel: MLModel
    public let visionProjectorModel: MLModel
    public let visionPositionEmbeddings: [Float]
    public let prefillModels: [MLModel]
    public let decodeModels: [MLModel]

    public static func load(
        rootURL: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> LFM2ModelBundle {
        let tokenizerFolderURL = rootURL
        let coreMLFolderURL = rootURL.appending(path: "CoreMLModels", directoryHint: .isDirectory)
        let metadataURL = coreMLFolderURL.appending(path: "meta.json")
        let metadataData = try Data(contentsOf: metadataURL)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let metadata = try decoder.decode(LFM2Metadata.self, from: metadataData)
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolderURL)

        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits

        func loadModel(_ fileName: String) throws -> MLModel {
            let sourceURL = coreMLFolderURL.appending(path: fileName)
            let compiledURL = try MLModel.compileModel(at: sourceURL)
            return try MLModel(contentsOf: compiledURL, configuration: configuration)
        }

        func loadFloat16File(_ fileName: String) throws -> [Float] {
            let data = try Data(contentsOf: coreMLFolderURL.appending(path: fileName))
            let count = data.count / MemoryLayout<UInt16>.size
            return data.withUnsafeBytes { rawBuffer in
                let values = rawBuffer.bindMemory(to: UInt16.self)
                return (0..<count).map { Float(Float16(bitPattern: UInt16(littleEndian: values[$0]))) }
            }
        }

        return LFM2ModelBundle(
            rootURL: rootURL,
            tokenizerFolderURL: tokenizerFolderURL,
            coreMLFolderURL: coreMLFolderURL,
            metadata: metadata,
            tokenizer: tokenizer,
            embeddingsModel: try loadModel(metadata.embeddingsModel),
            lmHeadModel: try loadModel(metadata.lmHeadModel),
            visionPatchEmbeddingModel: try loadModel(metadata.visionPatchEmbeddingModel),
            visionEncoderModel: try loadModel(metadata.visionEncoderModel),
            visionProjectorModel: try loadModel(metadata.visionProjectorModel),
            visionPositionEmbeddings: try loadFloat16File(metadata.visionPositionEmbeddingsFile),
            prefillModels: try metadata.prefillModels.map(loadModel),
            decodeModels: try metadata.decodeModels.map(loadModel)
        )
    }
}
