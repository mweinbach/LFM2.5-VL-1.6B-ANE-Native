import Testing
@testable import LFM2CoreML

@Test func chatFormatterRendersAssistantPrompt() throws {
    let rendered = try ChatFormatter.render(
        messages: [
            ChatMessage(role: "system", content: [.text("You are helpful.")]),
            ChatMessage(role: "user", content: [.text("Hello")])
        ]
    )

    #expect(rendered.hasPrefix("<|startoftext|>"))
    #expect(rendered.contains("<|im_start|>system\nYou are helpful.<|im_end|>\n"))
    #expect(rendered.hasSuffix("<|im_start|>assistant\n"))
}
