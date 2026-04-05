import Foundation

public enum ChatContent: Sendable, Hashable {
    case text(String)
    case image(URL)
    case imagePlaceholder
}

public struct ChatMessage: Sendable, Hashable {
    public let role: String
    public let content: [ChatContent]

    public init(role: String, content: [ChatContent]) {
        self.role = role
        self.content = content
    }
}

public enum ChatFormatterError: Error {
    case emptyConversation
}

public enum ChatFormatter {
    public static func render(messages: [ChatMessage], addGenerationPrompt: Bool = true, bosToken: String = "<|startoftext|>") throws -> String {
        guard !messages.isEmpty else {
            throw ChatFormatterError.emptyConversation
        }

        let lastAssistantIndex = messages.lastIndex { $0.role == "assistant" }
        var rendered = bosToken

        for (index, message) in messages.enumerated() {
            rendered += "<|im_start|>\(message.role)\n"
            var content = message.content.map(renderContent).joined()
            if message.role == "assistant", let lastAssistantIndex, index != lastAssistantIndex {
                content = stripPastThinking(content)
            }
            rendered += content
            rendered += "<|im_end|>\n"
        }

        if addGenerationPrompt {
            rendered += "<|im_start|>assistant\n"
        }
        return rendered
    }

    private static func renderContent(_ content: ChatContent) -> String {
        switch content {
        case let .text(text):
            return text
        case .image, .imagePlaceholder:
            return "<image>"
        }
    }

    private static func stripPastThinking(_ content: String) -> String {
        guard let range = content.range(of: "</think>", options: .backwards) else {
            return content
        }
        return String(content[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
