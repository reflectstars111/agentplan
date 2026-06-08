/// Settings screen — backend connection, model, theme preferences.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/backend_process.dart';
import '../providers/backend_provider.dart';
import '../providers/query_provider.dart';

class SettingsScreen extends ConsumerWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final backend = ref.watch(backendProcessProvider);
    final state = ref.watch(backendStateProvider).valueOrNull;
    final model = ref.watch(selectedModelProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // ── Backend Status ────────────────────────
          _sectionHeader('Backend'),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                children: [
                  _settingRow('Status', _statusText(state)),
                  _settingRow('URL', backend.baseUrl),
                  _settingRow('State', state?.name ?? 'unknown'),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      FilledButton.icon(
                        onPressed:
                            state == BackendState.running ? null : () => backend.start(port: 8000),
                        icon: const Icon(Icons.play_arrow, size: 18),
                        label: const Text('Start'),
                      ),
                      const SizedBox(width: 8),
                      OutlinedButton.icon(
                        onPressed:
                            state == BackendState.running ? () => backend.stop() : null,
                        icon: const Icon(Icons.stop, size: 18),
                        label: const Text('Stop'),
                      ),
                      const SizedBox(width: 8),
                      OutlinedButton.icon(
                        onPressed: () => backend.isHealthy(),
                        icon: const Icon(Icons.refresh, size: 18),
                        label: const Text('Check'),
                      ),
                    ],
                  ),
                  if (state == BackendState.error) ...[
                    const SizedBox(height: 8),
                    Text(backend.errorMessage,
                        style: const TextStyle(color: Colors.red, fontSize: 12)),
                  ],
                ],
              ),
            ),
          ),

          const SizedBox(height: 24),

          // ── Model ────────────────────────────────
          _sectionHeader('Model'),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: DropdownButtonFormField<String>(
                initialValue: model,
                decoration: const InputDecoration(
                  labelText: 'Default Model',
                  border: OutlineInputBorder(),
                ),
                items: const [
                  DropdownMenuItem(value: 'deepseek-chat', child: Text('DeepSeek Chat (V3)')),
                  DropdownMenuItem(value: 'deepseek-v4-flash', child: Text('DeepSeek V4 Flash')),
                  DropdownMenuItem(value: 'deepseek-v4-pro', child: Text('DeepSeek V4 Pro')),
                  DropdownMenuItem(value: 'gpt-4o', child: Text('OpenAI GPT-4o')),
                  DropdownMenuItem(value: 'claude-sonnet-4-6', child: Text('Claude Sonnet 4.6')),
                ],
                onChanged: (value) {
                  if (value != null) {
                    ref.read(selectedModelProvider.notifier).state = value;
                  }
                },
              ),
            ),
          ),

          const SizedBox(height: 24),

          // ── Theme ────────────────────────────────
          _sectionHeader('Appearance'),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                children: [
                  SwitchListTile(
                    title: const Text('Dark Mode'),
                    subtitle: const Text('Always enabled (light theme coming in Phase 2)'),
                    value: true,
                    onChanged: (_) {},
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 24),

          // ── About ────────────────────────────────
          _sectionHeader('About'),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Agent-OS Desktop Console',
                      style: TextStyle(fontWeight: FontWeight.w600)),
                  const SizedBox(height: 4),
                  const Text('Version 1.0.0',
                      style: TextStyle(color: Colors.grey, fontSize: 13)),
                  const SizedBox(height: 4),
                  Text(
                    'Von Neumann-inspired Multi-Agent Runtime\n'
                    'Multi-level memory • Hybrid retrieval • Task graph execution',
                    style: TextStyle(color: Colors.grey.shade600, fontSize: 12),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Text(title,
          style: const TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: Colors.grey,
              letterSpacing: 1)),
    );
  }

  Widget _settingRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 80,
            child: Text(label,
                style: const TextStyle(color: Colors.grey, fontSize: 13)),
          ),
          Expanded(
            child: Text(value,
                style: const TextStyle(
                    fontFamily: 'monospace', fontSize: 13),
                overflow: TextOverflow.ellipsis),
          ),
        ],
      ),
    );
  }

  String _statusText(BackendState? state) {
    switch (state) {
      case BackendState.running:
        return '● Connected';
      case BackendState.starting:
        return '◐ Starting...';
      case BackendState.error:
        return '✗ Error';
      case BackendState.stopped:
        return '○ Stopped';
      default:
        return '○ Stopped';
    }
  }
}
