/// Query input — mode toggle, model selector, text area, submit button.
library;

import 'package:flutter/material.dart';
import 'model_selector.dart';
import 'mode_toggle.dart';

class QueryInput extends StatelessWidget {
  final TextEditingController controller;
  final VoidCallback onSubmit;
  final bool isLoading;

  const QueryInput({
    super.key,
    required this.controller,
    required this.onSubmit,
    required this.isLoading,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface.withAlpha(80),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const ModeToggle(),
              const Spacer(),
              const ModelSelector(),
            ],
          ),
          const SizedBox(height: 8),
          TextField(
            controller: controller,
            maxLines: 4,
            minLines: 2,
            enabled: !isLoading,
            onSubmitted: (_) => onSubmit(),
            decoration: const InputDecoration(
              hintText: 'Ask a question or describe a complex task...',
              alignLabelWithHint: true,
            ),
            style: const TextStyle(fontSize: 14),
          ),
          const SizedBox(height: 8),
          Align(
            alignment: Alignment.centerRight,
            child: FilledButton.icon(
              onPressed: isLoading ? null : onSubmit,
              icon: isLoading
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.white),
                    )
                  : const Icon(Icons.send, size: 18),
              label: Text(isLoading ? 'Processing...' : 'Submit'),
            ),
          ),
        ],
      ),
    );
  }
}
