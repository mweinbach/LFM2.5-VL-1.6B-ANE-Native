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
        let metadata = try JSONDecoder().decode(LFM2Metadata.self, from: metadataData)
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolderURL)

        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits

        func loadModel(_ fileName: String) throws -> MLModel {
            try MLModel(contentsOf: coreMLFolderURL.appending(path: fileName), configuration: configuration)
        }

        return LFM2ModelBundle(
            rootURL: rootURL,
            tokenizerFolderURL: tokenizerFolderURL,
            coreMLFolderURL: coreMLFolderURL,
            metadata: metadata,
            tokenizer: tokenizer,
            embeddingsModel: try loadModel(metadata.embeddingsModel),
            lmHeadModel: try loadModel(metadata.lmHeadModel),
            prefillModels: try metadata.prefillModels.map(loadModel),
            decodeModels: try metadata.decodeModels.map(loadModel)
        )
    }
}
