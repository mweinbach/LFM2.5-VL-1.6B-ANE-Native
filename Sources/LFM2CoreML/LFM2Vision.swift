import CoreGraphics
import CoreML
import Foundation
import ImageIO

struct PreparedPrompt {
    let renderedPrompt: String
    let projectedImageEmbeddings: [Float]
}

struct ProcessedImageCrop {
    let patchValues: [Float]
    let patchRows: Int
    let patchCols: Int
    let projectedTokenCount: Int
}

struct ProcessedImageInfo {
    let placeholderExpansion: String
    let crops: [ProcessedImageCrop]
}

enum LFM2VisionError: Error {
    case imageLoadFailed(URL)
    case imageDecodeFailed(URL)
    case unsupportedImageCount(expected: Int, actual: Int)
    case imageEmbeddingCountMismatch(expected: Int, actual: Int)
}

final class LFM2VisionProcessor {
    private let metadata: LFM2Metadata
    private var positionEmbeddingCache: [String: [Float]] = [:]

    init(metadata: LFM2Metadata) {
        self.metadata = metadata
    }

    func prepare(messages: [ChatMessage], bundle: LFM2ModelBundle) throws -> PreparedPrompt {
        let rendered = try ChatFormatter.render(messages: messages, addGenerationPrompt: true)
        let imageURLs = collectImageURLs(messages: messages)
        guard !imageURLs.isEmpty else {
            return PreparedPrompt(renderedPrompt: rendered, projectedImageEmbeddings: [])
        }

        let processedImages = try imageURLs.map(processImage)
        let expandedPrompt = try expandPrompt(rendered, processedImages: processedImages)
        let embeddings = try projectImages(processedImages, bundle: bundle)
        return PreparedPrompt(renderedPrompt: expandedPrompt, projectedImageEmbeddings: embeddings)
    }

    private func collectImageURLs(messages: [ChatMessage]) -> [URL] {
        messages.flatMap { message in
            message.content.compactMap { content in
                if case let .image(url) = content {
                    return url
                }
                return nil
            }
        }
    }

    private func expandPrompt(_ rendered: String, processedImages: [ProcessedImageInfo]) throws -> String {
        let parts = rendered.components(separatedBy: "<image>")
        let placeholderCount = max(parts.count - 1, 0)
        guard placeholderCount == processedImages.count else {
            throw LFM2VisionError.unsupportedImageCount(expected: placeholderCount, actual: processedImages.count)
        }
        var output = parts[0]
        for index in 0..<processedImages.count {
            output += processedImages[index].placeholderExpansion
            output += parts[index + 1]
        }
        return output
    }

    private func projectImages(_ processedImages: [ProcessedImageInfo], bundle: LFM2ModelBundle) throws -> [Float] {
        let allCrops = processedImages.flatMap(\.crops)
        guard !allCrops.isEmpty else { return [] }

        let patchDim = 3 * metadata.encoderPatchSize * metadata.encoderPatchSize
        var pixelValues: [Float] = []
        pixelValues.reserveCapacity(allCrops.count * metadata.maxNumPatches * patchDim)
        for crop in allCrops {
            let validPatches = crop.patchRows * crop.patchCols
            pixelValues.append(contentsOf: crop.patchValues)
            if validPatches < metadata.maxNumPatches {
                pixelValues.append(contentsOf: Array(repeating: 0, count: (metadata.maxNumPatches - validPatches) * patchDim))
            }
        }

        let patchEmbeddingOutputs = try predict(
            model: bundle.visionPatchEmbeddingModel,
            inputs: [
                "pixel_values": try TensorHelpers.makeFloat16Array(
                    shape: [allCrops.count, metadata.maxNumPatches, patchDim],
                    values: pixelValues
                )
            ]
        )
        let patchEmbeddings = try multiArray(patchEmbeddingOutputs, name: "patch_embeddings")
        let flattenedPatchEmbeddings = TensorHelpers.flatten(patchEmbeddings)
        let patchEmbeddingStride = metadata.maxNumPatches * metadata.visionHiddenSize

        var projectedEmbeddings: [Float] = []
        for (cropIndex, crop) in allCrops.enumerated() {
            let validPatchCount = crop.patchRows * crop.patchCols
            let cropStart = cropIndex * patchEmbeddingStride
            var cropHiddenStates = Array(flattenedPatchEmbeddings[cropStart..<(cropStart + patchEmbeddingStride)])
            let resizedPositions = resizedPositionEmbeddings(rows: crop.patchRows, cols: crop.patchCols, bundle: bundle)
            for patchIndex in 0..<validPatchCount {
                let base = patchIndex * metadata.visionHiddenSize
                for channel in 0..<metadata.visionHiddenSize {
                    cropHiddenStates[base + channel] += resizedPositions[base + channel]
                }
            }

            let encoderOutputs = try predict(
                model: bundle.visionEncoderModel,
                inputs: [
                    "hidden_states": try TensorHelpers.makeFloat16Array(
                        shape: [1, metadata.maxNumPatches, metadata.visionHiddenSize],
                        values: cropHiddenStates
                    ),
                    "pixel_attention_mask": try TensorHelpers.makeInt32Array(
                        shape: [1, metadata.maxNumPatches],
                        values: Array(repeating: 1, count: validPatchCount) + Array(repeating: 0, count: metadata.maxNumPatches - validPatchCount)
                    )
                ]
            )
            let encodedStates = try multiArray(encoderOutputs, name: "last_hidden_state")
            let flattenedEncodedStates = TensorHelpers.flatten(encodedStates)
            let validHidden = Array(flattenedEncodedStates[0..<(validPatchCount * metadata.visionHiddenSize)])
            let projectorInput = try TensorHelpers.makeFloat16Array(
                shape: [1, crop.patchRows, crop.patchCols, metadata.visionHiddenSize],
                values: validHidden
            )
            let projected = try predict(model: bundle.visionProjectorModel, inputs: ["image_features": projectorInput])
            let projectedArray = try multiArray(projected, name: "projected_embeddings")
            projectedEmbeddings.append(contentsOf: TensorHelpers.flatten(projectedArray))
        }

        let expectedTokens = processedImages.flatMap(\.crops).reduce(0) { $0 + $1.projectedTokenCount }
        let actualTokens = projectedEmbeddings.count / metadata.hiddenSize
        guard expectedTokens == actualTokens else {
            throw LFM2VisionError.imageEmbeddingCountMismatch(expected: expectedTokens, actual: actualTokens)
        }
        return projectedEmbeddings
    }

    private func resizedPositionEmbeddings(rows: Int, cols: Int, bundle: LFM2ModelBundle) -> [Float] {
        let cacheKey = "\(rows)x\(cols)"
        if let cached = positionEmbeddingCache[cacheKey] {
            return cached
        }

        let sourceSize = metadata.visionPositionEmbeddingSize
        let hiddenSize = metadata.visionHiddenSize
        let source = bundle.visionPositionEmbeddings
        let scaleY = Double(sourceSize) / Double(rows)
        let scaleX = Double(sourceSize) / Double(cols)
        var output = Array(repeating: Float(0), count: rows * cols * hiddenSize)

        for row in 0..<rows {
            let srcY = (Double(row) + 0.5) * scaleY - 0.5
            let y0 = max(Int(floor(srcY)), 0)
            let y1 = min(y0 + 1, sourceSize - 1)
            let wy = Float(max(0, min(1, srcY - Double(y0))))
            let wy0 = 1 - wy
            let wy1 = wy
            for col in 0..<cols {
                let srcX = (Double(col) + 0.5) * scaleX - 0.5
                let x0 = max(Int(floor(srcX)), 0)
                let x1 = min(x0 + 1, sourceSize - 1)
                let wx = Float(max(0, min(1, srcX - Double(x0))))
                let wx0 = 1 - wx
                let wx1 = wx

                let topLeft = (y0 * sourceSize + x0) * hiddenSize
                let topRight = (y0 * sourceSize + x1) * hiddenSize
                let bottomLeft = (y1 * sourceSize + x0) * hiddenSize
                let bottomRight = (y1 * sourceSize + x1) * hiddenSize
                let destination = (row * cols + col) * hiddenSize

                for channel in 0..<hiddenSize {
                    output[destination + channel] =
                        source[topLeft + channel] * wy0 * wx0 +
                        source[topRight + channel] * wy0 * wx1 +
                        source[bottomLeft + channel] * wy1 * wx0 +
                        source[bottomRight + channel] * wy1 * wx1
                }
            }
        }

        positionEmbeddingCache[cacheKey] = output
        return output
    }

    private func predict(model: MLModel, inputs: [String: Any]) throws -> MLFeatureProvider {
        try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: inputs))
    }

    private func multiArray(_ outputs: MLFeatureProvider, name: String) throws -> MLMultiArray {
        guard let value = outputs.featureValue(for: name)?.multiArrayValue else {
            throw LFM2PipelineError.invalidModelOutput(name)
        }
        return value
    }

    private func processImage(url: URL) throws -> ProcessedImageInfo {
        let cgImage = try loadImage(url: url)
        let width = cgImage.width
        let height = cgImage.height

        let totalFactor = metadata.encoderPatchSize * metadata.downsampleFactor
        let smartSize = smartResize(height: height, width: width, totalFactor: totalFactor)
        let tooLarge = isImageTooLarge(height: height, width: width, totalFactor: totalFactor)

        if tooLarge {
            let layout = gridLayout(height: height, width: width)
            let resized = try resize(image: cgImage, width: layout.targetWidth, height: layout.targetHeight)
            let tiles = splitIntoTiles(image: resized, rows: layout.gridHeight, cols: layout.gridWidth)
            var crops = tiles.map { tile in
                makeCrop(tile, resizedHeight: metadata.tileSize, resizedWidth: metadata.tileSize)
            }
            if metadata.useThumbnail {
                let thumbnail = try resize(image: cgImage, width: smartSize.width, height: smartSize.height)
                crops.append(makeCrop(thumbnail, resizedHeight: smartSize.height, resizedWidth: smartSize.width))
            }
            return ProcessedImageInfo(
                placeholderExpansion: buildPlaceholder(rows: layout.gridHeight, cols: layout.gridWidth, imageHeight: smartSize.height, imageWidth: smartSize.width),
                crops: crops
            )
        }

        let resized = try resize(image: cgImage, width: smartSize.width, height: smartSize.height)
        return ProcessedImageInfo(
            placeholderExpansion: buildPlaceholder(rows: 1, cols: 1, imageHeight: smartSize.height, imageWidth: smartSize.width),
            crops: [makeCrop(resized, resizedHeight: smartSize.height, resizedWidth: smartSize.width)]
        )
    }

    private func buildPlaceholder(rows: Int, cols: Int, imageHeight: Int, imageWidth: Int) -> String {
        let tokensPerTile = projectedTokenCount(height: metadata.tileSize, width: metadata.tileSize)
        let tokensForImage = projectedTokenCount(height: imageHeight, width: imageWidth)
        var parts: [String] = ["<|image_start|>"]
        if rows == 1, cols == 1 {
            parts.append(String(repeating: "<image>", count: tokensForImage))
        } else {
            for row in 0..<rows {
                for col in 0..<cols {
                    parts.append("<|img_row_\(row + 1)_col_\(col + 1)|>")
                    parts.append(String(repeating: "<image>", count: tokensPerTile))
                }
            }
            if metadata.useThumbnail {
                parts.append("<|img_thumbnail|>")
                parts.append(String(repeating: "<image>", count: tokensForImage))
            }
        }
        parts.append("<|image_end|>")
        return parts.joined()
    }

    private func projectedTokenCount(height: Int, width: Int) -> Int {
        let patchRows = height / metadata.encoderPatchSize
        let patchCols = width / metadata.encoderPatchSize
        let tokenRows = Int(ceil(Double(patchRows) / Double(metadata.downsampleFactor)))
        let tokenCols = Int(ceil(Double(patchCols) / Double(metadata.downsampleFactor)))
        return tokenRows * tokenCols
    }

    private func makeCrop(_ image: CGImage, resizedHeight: Int, resizedWidth: Int) -> ProcessedImageCrop {
        let patchRows = resizedHeight / metadata.encoderPatchSize
        let patchCols = resizedWidth / metadata.encoderPatchSize
        return ProcessedImageCrop(
            patchValues: patchify(image: image),
            patchRows: patchRows,
            patchCols: patchCols,
            projectedTokenCount: projectedTokenCount(height: resizedHeight, width: resizedWidth)
        )
    }

    private func patchify(image: CGImage) -> [Float] {
        let width = image.width
        let height = image.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var data = [UInt8](repeating: 0, count: height * bytesPerRow)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        guard let context = CGContext(
            data: &data,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            return []
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let patchSize = metadata.encoderPatchSize
        let patchDim = 3 * patchSize * patchSize
        var output: [Float] = []
        output.reserveCapacity((width / patchSize) * (height / patchSize) * patchDim)
        for patchY in stride(from: 0, to: height, by: patchSize) {
            for patchX in stride(from: 0, to: width, by: patchSize) {
                for y in 0..<patchSize {
                    for x in 0..<patchSize {
                        let offset = ((patchY + y) * bytesPerRow) + ((patchX + x) * bytesPerPixel)
                        let red = (Float(data[offset]) / 255.0 - 0.5) / 0.5
                        let green = (Float(data[offset + 1]) / 255.0 - 0.5) / 0.5
                        let blue = (Float(data[offset + 2]) / 255.0 - 0.5) / 0.5
                        output.append(red)
                        output.append(green)
                        output.append(blue)
                    }
                }
            }
        }
        return output
    }

    private func loadImage(url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw LFM2VisionError.imageLoadFailed(url)
        }
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw LFM2VisionError.imageDecodeFailed(url)
        }
        return image
    }

    private func resize(image: CGImage, width: Int, height: Int) throws -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = width * 4
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            throw LFM2VisionError.imageDecodeFailed(URL(fileURLWithPath: "/"))
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let resized = context.makeImage() else {
            throw LFM2VisionError.imageDecodeFailed(URL(fileURLWithPath: "/"))
        }
        return resized
    }

    private func splitIntoTiles(image: CGImage, rows: Int, cols: Int) -> [CGImage] {
        let tileWidth = image.width / cols
        let tileHeight = image.height / rows
        var tiles: [CGImage] = []
        tiles.reserveCapacity(rows * cols)
        for row in 0..<rows {
            for col in 0..<cols {
                let rect = CGRect(x: col * tileWidth, y: row * tileHeight, width: tileWidth, height: tileHeight)
                if let tile = image.cropping(to: rect) {
                    tiles.append(tile)
                }
            }
        }
        return tiles
    }

    private func isImageTooLarge(height: Int, width: Int, totalFactor: Int) -> Bool {
        let roundedHeight = max(metadata.encoderPatchSize, roundedByFactor(height, factor: totalFactor))
        let roundedWidth = max(metadata.encoderPatchSize, roundedByFactor(width, factor: totalFactor))
        let maxPixelsTolerance = 2.0
        let threshold = Double(metadata.maxImageTokens * metadata.encoderPatchSize * metadata.encoderPatchSize * metadata.downsampleFactor * metadata.downsampleFactor) * maxPixelsTolerance
        return Double(roundedHeight * roundedWidth) > threshold
    }

    private func smartResize(height: Int, width: Int, totalFactor: Int) -> (width: Int, height: Int) {
        let minPixels = metadata.minImageTokens * metadata.encoderPatchSize * metadata.encoderPatchSize * metadata.downsampleFactor * metadata.downsampleFactor
        let maxPixels = metadata.maxImageTokens * metadata.encoderPatchSize * metadata.encoderPatchSize * metadata.downsampleFactor * metadata.downsampleFactor
        var resizedHeight = max(totalFactor, roundedByFactor(height, factor: totalFactor))
        var resizedWidth = max(totalFactor, roundedByFactor(width, factor: totalFactor))

        if resizedHeight * resizedWidth > maxPixels {
            let beta = sqrt(Double(height * width) / Double(maxPixels))
            resizedHeight = max(totalFactor, Int(floor(Double(height) / beta / Double(totalFactor))) * totalFactor)
            resizedWidth = max(totalFactor, Int(floor(Double(width) / beta / Double(totalFactor))) * totalFactor)
        } else if resizedHeight * resizedWidth < minPixels {
            let beta = sqrt(Double(minPixels) / Double(height * width))
            resizedHeight = Int(ceil(Double(height) * beta / Double(totalFactor))) * totalFactor
            resizedWidth = Int(ceil(Double(width) * beta / Double(totalFactor))) * totalFactor
        }
        return (resizedWidth, resizedHeight)
    }

    private func roundedByFactor(_ value: Int, factor: Int) -> Int {
        Int((Double(value) / Double(factor)).rounded()) * factor
    }

    private func gridLayout(height: Int, width: Int) -> (gridWidth: Int, gridHeight: Int, targetWidth: Int, targetHeight: Int) {
        let aspectRatio = Double(width) / Double(height)
        let ratios = targetRatios(minTiles: metadata.minTiles, maxTiles: metadata.maxTiles)
        var bestRatio = (1, 1)
        var bestDiff = Double.greatestFiniteMagnitude
        let area = width * height
        for ratio in ratios {
            let targetAspectRatio = Double(ratio.0) / Double(ratio.1)
            let diff = abs(aspectRatio - targetAspectRatio)
            if diff < bestDiff {
                bestDiff = diff
                bestRatio = ratio
            } else if diff == bestDiff {
                let targetArea = metadata.tileSize * metadata.tileSize * ratio.0 * ratio.1
                if area > targetArea / 2 {
                    bestRatio = ratio
                }
            }
        }
        return (
            gridWidth: bestRatio.0,
            gridHeight: bestRatio.1,
            targetWidth: metadata.tileSize * bestRatio.0,
            targetHeight: metadata.tileSize * bestRatio.1
        )
    }

    private func targetRatios(minTiles: Int, maxTiles: Int) -> [(Int, Int)] {
        var ratios = Set<[Int]>()
        for n in minTiles...maxTiles {
            for width in 1...n {
                for height in 1...n where minTiles...maxTiles ~= width * height {
                    ratios.insert([width, height])
                }
            }
        }
        return ratios.map { ($0[0], $0[1]) }.sorted { ($0.0 * $0.1) < ($1.0 * $1.1) }
    }
}
