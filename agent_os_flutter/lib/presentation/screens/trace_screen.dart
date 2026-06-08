/// Trace detail screen — full trace timeline view.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../data/models/trace.dart';
import '../providers/backend_provider.dart';

class TraceScreen extends ConsumerStatefulWidget {
  final String traceId;
  const TraceScreen({super.key, required this.traceId});

  @override
  ConsumerState<TraceScreen> createState() => _TraceScreenState();
}

class _TraceScreenState extends ConsumerState<TraceScreen> {
  TraceData? _trace;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadTrace();
  }

  Future<void> _loadTrace() async {
    try {
      final client = ref.read(agentApiClientProvider);
      final trace = await client.getTrace(widget.traceId);
      setState(() {
        _trace = trace;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Trace: ${widget.traceId.length > 24
            ? '${widget.traceId.substring(0, 24)}...'
            : widget.traceId}'),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator(strokeWidth: 2))
          : _error != null
              ? Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.error_outline, size: 48, color: Colors.red),
                      const SizedBox(height: 12),
                      Text(_error!, style: const TextStyle(color: Colors.red)),
                      const SizedBox(height: 12),
                      OutlinedButton(
                          onPressed: _loadTrace, child: const Text('Retry')),
                    ],
                  ),
                )
              : _buildTrace(),
    );
  }

  Widget _buildTrace() {
    if (_trace == null || _trace!.steps.isEmpty) {
      return const Center(child: Text('No trace steps found'));
    }

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: _trace!.steps.length,
      itemBuilder: (context, index) {
        final step = _trace!.steps[index];
        return _buildStep(step, index);
      },
    );
  }

  Widget _buildStep(TraceStep step, int index) {
    final isSuccess = step.status == 'success';
    final isFailed = step.status == 'failed';
    final isLast = index == _trace!.steps.length - 1;

    return IntrinsicHeight(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Timeline connector
          SizedBox(
            width: 32,
            child: Column(
              children: [
                Container(
                  width: 12,
                  height: 12,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: isSuccess
                        ? Colors.green
                        : isFailed
                            ? Colors.red
                            : Colors.grey,
                    border: Border.all(
                        color: isSuccess ? Colors.greenAccent : Colors.grey,
                        width: 2),
                  ),
                ),
                if (!isLast)
                  Container(
                    width: 2,
                    height: 60,
                    color: Colors.grey.shade700,
                  ),
              ],
            ),
          ),

          // Step content
          Expanded(
            child: Padding(
              padding: const EdgeInsets.only(bottom: 16),
              child: Card(
                color: isFailed ? Colors.red.withAlpha(15) : null,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 6, vertical: 2),
                            decoration: BoxDecoration(
                              color: Colors.cyan.withAlpha(25),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: Text(step.type,
                                style: const TextStyle(
                                    color: Colors.cyan,
                                    fontSize: 12,
                                    fontWeight: FontWeight.w600)),
                          ),
                          const Spacer(),
                          Icon(
                            isSuccess
                                ? Icons.check
                                : isFailed
                                    ? Icons.close
                                    : Icons.remove,
                            size: 16,
                            color: isSuccess
                                ? Colors.green
                                : isFailed
                                    ? Colors.red
                                    : Colors.grey,
                          ),
                        ],
                      ),
                      if (step.error != null && step.error!.isNotEmpty) ...[
                        const SizedBox(height: 6),
                        Text(step.error!,
                            style: const TextStyle(
                                color: Colors.red, fontSize: 12)),
                      ],
                      if (step.input.isNotEmpty) ...[
                        const SizedBox(height: 4),
                        Text(
                          'Input: ${_truncate(step.input.toString(), 100)}',
                          style: const TextStyle(
                              color: Colors.grey, fontSize: 11),
                        ),
                      ],
                      if (step.output.isNotEmpty) ...[
                        const SizedBox(height: 2),
                        Text(
                          'Output: ${_truncate(step.output.toString(), 100)}',
                          style: const TextStyle(
                              color: Colors.grey, fontSize: 11),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _truncate(String text, int maxLen) {
    if (text.length <= maxLen) return text;
    return '${text.substring(0, maxLen)}...';
  }
}
