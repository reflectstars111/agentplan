/// Response view — displays agent response with optional error styling.
library;

import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';

class ResponseView extends StatelessWidget {
  final String response;
  final bool isError;

  const ResponseView({
    super.key,
    required this.response,
    this.isError = false,
  });

  @override
  Widget build(BuildContext context) {
    if (response.isEmpty) return const SizedBox.shrink();

    return Card(
      color: isError ? Colors.red.withAlpha(10) : null,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: isError
            ? Text(response,
                style: const TextStyle(
                    color: Colors.red, fontFamily: 'monospace', fontSize: 13))
            : MarkdownBody(
                data: response,
                selectable: true,
                styleSheet: MarkdownStyleSheet(
                  p: const TextStyle(fontSize: 14, height: 1.5, color: Color(0xFFE0E0E0)),
                  code: TextStyle(
                    fontSize: 12,
                    color: Colors.amber.shade200,
                    backgroundColor: const Color(0xFF0A0A1A),
                    fontFamily: 'monospace',
                  ),
                  codeblockDecoration: BoxDecoration(
                    color: const Color(0xFF0A0A1A),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: const Color(0xFF333355)),
                  ),
                  h1: const TextStyle(color: Color(0xFF00D4FF), fontSize: 20),
                  h2: const TextStyle(color: Color(0xFF00D4FF), fontSize: 17),
                  h3: const TextStyle(color: Color(0xFF00D4FF), fontSize: 15),
                  blockquote: const TextStyle(
                    color: Color(0xFF888888),
                    fontStyle: FontStyle.italic,
                  ),
                  blockquoteDecoration: const BoxDecoration(
                    border: Border(
                      left: BorderSide(color: Color(0xFF00D4FF), width: 3),
                    ),
                  ),
                ),
              ),
      ),
    );
  }
}
