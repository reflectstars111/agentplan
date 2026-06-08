/// Connection indicator — shows backend health status in the app bar.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/backend_process.dart';
import '../providers/backend_provider.dart';

class ConnectionIndicator extends ConsumerWidget {
  const ConnectionIndicator({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(backendStateProvider).valueOrNull;

    final (icon, color, label) = switch (state) {
      BackendState.running => (Icons.circle, Colors.green, 'Connected'),
      BackendState.starting => (Icons.circle_outlined, Colors.amber, 'Starting...'),
      BackendState.error => (Icons.error_outline, Colors.red, 'Error'),
      _ => (Icons.circle_outlined, Colors.grey, 'Disconnected'),
    };

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8),
      child: Chip(
        avatar: Icon(icon, size: 10, color: color),
        label: Text(label, style: TextStyle(color: color, fontSize: 11)),
        backgroundColor: color.withAlpha(20),
        side: BorderSide.none,
        padding: EdgeInsets.zero,
        materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
      ),
    );
  }
}
