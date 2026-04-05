// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "LFM2CoreML",
    platforms: [
        .iOS(.v18),
        .macOS(.v15)
    ],
    products: [
        .library(name: "LFM2CoreML", targets: ["LFM2CoreML"]),
        .executable(name: "ANEInferenceCLI", targets: ["ANEInferenceCLI"])
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0")
    ],
    targets: [
        .target(
            name: "LFM2CoreML",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers")
            ]
        ),
        .executableTarget(
            name: "ANEInferenceCLI",
            dependencies: ["LFM2CoreML"]
        ),
        .testTarget(
            name: "LFM2CoreMLTests",
            dependencies: ["LFM2CoreML"]
        )
    ]
)
