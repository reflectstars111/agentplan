/// HomeScreen — main query interface with sidebar and response area.
///
/// Layout: Sidebar (uploads + conversation history) | Main (query + response + trace)
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/query_provider.dart';
import '../providers/conversation_provider.dart';
import '../widgets/sidebar.dart';
import '../widgets/query_input.dart';
import '../widgets/response_view.dart';
import '../widgets/trace_timeline.dart';
import '../widgets/connection_indicator.dart';

class HomeScreen extends ConsumerStatefulWidget {
  const HomeScreen({super.key});

  @override
  ConsumerState<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends ConsumerState<HomeScreen> {
  final _queryController = TextEditingController();
  int? _currentConversationId;

  @override
  void dispose() {
    _queryController.dispose();
    super.dispose();
  }

  Future<void> _submitQuery() async {
    final query = _queryController.text.trim();
    if (query.isEmpty) return;

    // Ensure we have a conversation
    if (_currentConversationId == null) {
      final title = query.length > 50 ? '${query.substring(0, 50)}...' : query;
      _currentConversationId = await ref
          .read(conversationNotifierProvider.notifier)
          .createConversation(title);
      ref.read(activeConversationIdProvider.notifier).state =
          _currentConversationId;
    }

    // Save user message locally
    await ref.read(conversationNotifierProvider.notifier).addMessage(
      _currentConversationId!,
      'user',
      query,
      model: ref.read(selectedModelProvider),
    );

    // Execute query
    await ref.read(queryNotifierProvider.notifier).executeQuery(query);

    final queryState = ref.read(queryNotifierProvider);
    if (queryState == QueryState.success) {
      final notifier = ref.read(queryNotifierProvider.notifier);
      // Save agent response locally
      await ref.read(conversationNotifierProvider.notifier).addMessage(
        _currentConversationId!,
        'agent',
        notifier.response,
        model: ref.read(selectedModelProvider),
        traceId: notifier.traceId,
        verified: notifier.verified,
      );
    }

    _queryController.clear();
  }

  void _newConversation() {
    _currentConversationId = null;
    ref.read(activeConversationIdProvider.notifier).state = null;
    ref.read(queryNotifierProvider.notifier).reset();
    _queryController.clear();
  }

  void _selectConversation(int id) {
    _currentConversationId = id;
    ref.read(activeConversationIdProvider.notifier).state = id;
  }

  @override
  Widget build(BuildContext context) {
    final queryState = ref.watch(queryNotifierProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Agent-OS Console'),
        actions: [
          const ConnectionIndicator(),
          IconButton(
            icon: const Icon(Icons.settings_outlined, size: 20),
            tooltip: 'Settings',
            onPressed: () => Navigator.pushNamed(context, '/settings'),
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: Row(
        children: [
          // ── Sidebar ─────────────────────────────────
          Sidebar(
            onNewConversation: _newConversation,
            onSelectConversation: _selectConversation,
            currentConversationId: _currentConversationId,
          ),

          const VerticalDivider(width: 1),

          // ── Main Content ────────────────────────────
          Expanded(
            child: Column(
              children: [
                // Query input area
                QueryInput(
                  controller: _queryController,
                  onSubmit: _submitQuery,
                  isLoading: queryState == QueryState.loading,
                ),
                const Divider(height: 1),

                // Response + Trace area
                Expanded(
                  child: _buildContent(queryState),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildContent(QueryState queryState) {
    if (queryState == QueryState.loading) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(strokeWidth: 2),
            SizedBox(height: 16),
            Text('Processing...', style: TextStyle(color: Colors.grey)),
          ],
        ),
      );
    }

    if (queryState == QueryState.idle) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.psychology_outlined,
                size: 64, color: Colors.grey.shade700),
            const SizedBox(height: 16),
            Text(
              'Ask a question or describe a task',
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    color: Colors.grey,
                  ),
            ),
            const SizedBox(height: 8),
            Text(
              'Upload knowledge first for document-aware answers',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: Colors.grey.shade600,
                  ),
            ),
          ],
        ),
      );
    }

    final notifier = ref.read(queryNotifierProvider.notifier);

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Status badges
          _buildStatusBadges(notifier),

          const SizedBox(height: 12),

          // Response content
          ResponseView(
            response: queryState == QueryState.error
                ? notifier.errorMessage
                : notifier.response,
            isError: queryState == QueryState.error,
          ),

          // Intent card
          if (notifier.intent != null) ...[
            const SizedBox(height: 16),
            _buildIntentCard(notifier.intent!),
          ],

          // Task graph card
          if (notifier.taskGraph != null && notifier.taskGraph!.isNotEmpty) ...[
            const SizedBox(height: 16),
            _buildTaskGraphCard(notifier.taskGraph!),
          ],

          // Writeback
          if (notifier.writeback.isNotEmpty) ...[
            const SizedBox(height: 16),
            _buildWritebackCard(notifier.writeback),
          ],

          // Conflicts
          if (notifier.conflicts.isNotEmpty) ...[
            const SizedBox(height: 16),
            _buildConflictsCard(notifier.conflicts),
          ],

          // Trace timeline
          if (notifier.traceData != null) ...[
            const SizedBox(height: 16),
            TraceTimeline(trace: notifier.traceData!),
          ],
        ],
      ),
    );
  }

  Widget _buildStatusBadges(QueryNotifier notifier) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        if (notifier.verified)
          Chip(
            avatar: const Icon(Icons.check_circle, size: 16, color: Colors.green),
            label: const Text('Verified'),
          ),
        if (notifier.traceId.isNotEmpty)
          Chip(
            avatar: const Icon(Icons.timeline, size: 16, color: Colors.cyan),
            label: Text(notifier.traceId.length > 20
                ? '${notifier.traceId.substring(0, 20)}...'
                : notifier.traceId),
          ),
      ],
    );
  }

  Widget _buildIntentCard(Map<String, dynamic> intent) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Intent',
                style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600,
                    color: Colors.grey, letterSpacing: 1)),
            const SizedBox(height: 6),
            Row(
              children: [
                Text('Type: ',
                    style: const TextStyle(color: Colors.grey, fontSize: 13)),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: Colors.cyan.withAlpha(25),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    intent['intent_type'] ?? 'unknown',
                    style: const TextStyle(
                        color: Colors.cyan,
                        fontSize: 13,
                        fontWeight: FontWeight.w600),
                  ),
                ),
                if (intent['confidence'] != null) ...[
                  const SizedBox(width: 12),
                  Text(
                    '${((intent['confidence'] as num) * 100).toStringAsFixed(0)}% confidence',
                    style: const TextStyle(color: Colors.grey, fontSize: 12),
                  ),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTaskGraphCard(Map<String, dynamic> graph) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Task Graph',
                style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600,
                    color: Colors.grey, letterSpacing: 1)),
            const SizedBox(height: 8),
            Row(
              children: [
                _metricChip('Nodes', '${graph['node_count'] ?? 0}', Colors.cyan),
                const SizedBox(width: 12),
                _metricChip('Done', '${graph['completed'] ?? 0}', Colors.green),
                const SizedBox(width: 12),
                _metricChip('Failed', '${graph['failed'] ?? 0}', Colors.red),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _metricChip(String label, String value, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(color: Colors.grey, fontSize: 11)),
        const SizedBox(height: 2),
        Text(value,
            style: TextStyle(
                color: color, fontSize: 20, fontWeight: FontWeight.bold)),
      ],
    );
  }

  Widget _buildWritebackCard(Map<String, dynamic> writeback) {
    final action = writeback['action'] ?? 'skip';
    final reason = writeback['reason'] ?? '';
    final score = (writeback['score'] as num?)?.toDouble() ?? 0.0;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Memory Writeback',
                style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600,
                    color: Colors.grey, letterSpacing: 1)),
            const SizedBox(height: 6),
            Row(
              children: [
                Icon(
                  action == 'write'
                      ? Icons.save
                      : action == 'ask_user'
                          ? Icons.help_outline
                          : Icons.skip_next,
                  size: 16,
                  color: Colors.amber,
                ),
                const SizedBox(width: 6),
                Text('Action: $action',
                    style: const TextStyle(color: Colors.amber, fontSize: 13)),
                const SizedBox(width: 16),
                Text('Score: ${score.toStringAsFixed(2)}',
                    style: const TextStyle(color: Colors.grey, fontSize: 12)),
              ],
            ),
            if (reason.isNotEmpty) ...[
              const SizedBox(height: 4),
              Text(reason,
                  style: const TextStyle(color: Colors.grey, fontSize: 12)),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildConflictsCard(List<String> conflicts) {
    return Card(
      color: Colors.red.withAlpha(15),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.warning_amber, size: 16, color: Colors.orange),
                SizedBox(width: 6),
                Text('Conflicts Detected',
                    style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600,
                        color: Colors.orange, letterSpacing: 1)),
              ],
            ),
            const SizedBox(height: 6),
            ...conflicts.map((c) => Padding(
                  padding: const EdgeInsets.only(bottom: 4),
                  child: Text('• $c',
                      style: const TextStyle(color: Colors.orange, fontSize: 12)),
                )),
          ],
        ),
      ),
    );
  }
}
