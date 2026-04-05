import Foundation

public struct LFM2ChunkMetadata: Codable, Sendable {
    public let name: String
    public let startLayer: Int
    public let endLayer: Int
    public let convLayerIndices: [Int]
    public let attentionLayerIndices: [Int]
}

public struct LFM2Metadata: Codable, Sendable {
    public let contextLength: Int
    public let prefillLength: Int
    public let batchSize: Int
    public let hiddenSize: Int
    public let vocabSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let convKernelSize: Int
    public let bosTokenId: Int
    public let eosTokenId: Int
    public let padTokenId: Int
    public let imageTokenId: Int
    public let imageStartTokenId: Int
    public let imageEndTokenId: Int
    public let chunkOrder: [String]
    public let chunks: [LFM2ChunkMetadata]
    public let layerTypes: [String]
    public let embeddingsModel: String
    public let lmHeadModel: String
    public let prefillModels: [String]
    public let decodeModels: [String]
    public let chatTemplate: String
}
