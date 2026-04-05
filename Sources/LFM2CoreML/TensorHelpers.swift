import CoreML
import Foundation

enum TensorHelpers {
    static func makeInt32Array(shape: [Int], values: [Int]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .int32)
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }

    static func makeFloat16Array(shape: [Int], repeatedValue: Float = 0) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float16)
        for index in 0..<array.count {
            array[index] = NSNumber(value: repeatedValue)
        }
        return array
    }

    static func makeFloat16Array(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float16)
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }

    static func flatten(_ array: MLMultiArray) -> [Float] {
        (0..<array.count).map { array[$0].floatValue }
    }

    static func lastTokenLogits(from array: MLMultiArray, vocabSize: Int) -> [Float] {
        let flat = flatten(array)
        guard flat.count >= vocabSize else { return flat }
        return Array(flat[(flat.count - vocabSize)..<flat.count])
    }
}
