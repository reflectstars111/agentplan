/// Model selector — dropdown for choosing LLM model.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/query_provider.dart';

class ModelSelector extends ConsumerWidget {
  const ModelSelector({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final model = ref.watch(selectedModelProvider);

    return SizedBox(
      width: 180,
      child: DropdownButtonFormField<String>(
        initialValue: model,
        isDense: true,
        decoration: const InputDecoration(
          labelText: 'Model',
          border: OutlineInputBorder(),
          contentPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        ),
        style: const TextStyle(fontSize: 12),
        items: const [
          DropdownMenuItem(value: 'deepseek-chat', child: Text('DeepSeek V3', style: TextStyle(fontSize: 12))),
          DropdownMenuItem(value: 'deepseek-v4-flash', child: Text('V4 Flash', style: TextStyle(fontSize: 12))),
          DropdownMenuItem(value: 'deepseek-v4-pro', child: Text('V4 Pro', style: TextStyle(fontSize: 12))),
          DropdownMenuItem(value: 'gpt-4o', child: Text('GPT-4o', style: TextStyle(fontSize: 12))),
          DropdownMenuItem(value: 'claude-sonnet-4-6', child: Text('Claude Sonnet 4.6', style: TextStyle(fontSize: 12))),
        ],
        onChanged: (value) {
          if (value != null) {
            ref.read(selectedModelProvider.notifier).state = value;
          }
        },
      ),
    );
  }
}
