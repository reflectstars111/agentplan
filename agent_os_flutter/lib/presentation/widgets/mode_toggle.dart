/// Mode toggle — segmented button for Simple vs Task query modes.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/query_provider.dart';

class ModeToggle extends ConsumerWidget {
  const ModeToggle({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final mode = ref.watch(queryModeProvider);

    return SegmentedButton<String>(
      segments: const [
        ButtonSegment(value: 'simple', label: Text('Simple')),
        ButtonSegment(value: 'task', label: Text('Task Graph')),
      ],
      selected: {mode},
      onSelectionChanged: (selected) {
        ref.read(queryModeProvider.notifier).state = selected.first;
      },
      style: SegmentedButton.styleFrom(
        selectedBackgroundColor: Theme.of(context).colorScheme.primary,
        selectedForegroundColor: Theme.of(context).colorScheme.onPrimary,
      ),
    );
  }
}
