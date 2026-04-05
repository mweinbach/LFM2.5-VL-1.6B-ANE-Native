import Foundation
import LFM2CoreML

struct CLIArguments {
    var bundleRoot: URL
    var prompt: String
    var systemPrompt: String?
    var maxNewTokens: Int = 128
    var temperature: Float = 0
    var topK: Int = 40
}

enum CLIError: Error {
    case missingPrompt
    case invalidValue(String)
}

func parseArguments() throws -> CLIArguments {
    var bundleRoot = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
    var prompt: String?
    var systemPrompt: String?
    var maxNewTokens = 128
    var temperature: Float = 0
    var topK = 40

    var index = 1
    let args = CommandLine.arguments
    while index < args.count {
        let arg = args[index]
        switch arg {
        case "--bundle-root":
            index += 1
            bundleRoot = URL(fileURLWithPath: args[index], isDirectory: true)
        case "--prompt":
            index += 1
            prompt = args[index]
        case "--system":
            index += 1
            systemPrompt = args[index]
        case "--max-new-tokens":
            index += 1
            guard let value = Int(args[index]) else { throw CLIError.invalidValue(arg) }
            maxNewTokens = value
        case "--temperature":
            index += 1
            guard let value = Float(args[index]) else { throw CLIError.invalidValue(arg) }
            temperature = value
        case "--top-k":
            index += 1
            guard let value = Int(args[index]) else { throw CLIError.invalidValue(arg) }
            topK = value
        case "--help", "-h":
            print("Usage: swift run ANEInferenceCLI --prompt \"Hello\" [--system \"You are helpful\"] [--bundle-root /path/to/ANE_Native] [--max-new-tokens 128] [--temperature 0] [--top-k 40]")
            Foundation.exit(0)
        default:
            throw CLIError.invalidValue(arg)
        }
        index += 1
    }

    guard let prompt else {
        throw CLIError.missingPrompt
    }

    return CLIArguments(
        bundleRoot: bundleRoot,
        prompt: prompt,
        systemPrompt: systemPrompt,
        maxNewTokens: maxNewTokens,
        temperature: temperature,
        topK: topK
    )
}

@main
struct ANEInferenceCLI {
    static func main() async throws {
        do {
            let arguments = try parseArguments()
            let bundle = try await LFM2ModelBundle.load(rootURL: arguments.bundleRoot)
            let pipeline = try LFM2Pipeline(bundle: bundle)
            var messages: [ChatMessage] = []
            if let systemPrompt = arguments.systemPrompt {
                messages.append(ChatMessage(role: "system", content: [.text(systemPrompt)]))
            }
            messages.append(ChatMessage(role: "user", content: [.text(arguments.prompt)]))
            let output = try await pipeline.generate(
                messages: messages,
                config: GenerationConfig(
                    maxNewTokens: arguments.maxNewTokens,
                    temperature: arguments.temperature,
                    topK: arguments.topK
                )
            )
            print(output)
        } catch {
            fputs("ANEInferenceCLI error: \(error)\n", stderr)
            Foundation.exit(1)
        }
    }
}
