/// Trace timeline — vertical timeline of execution steps.
library;

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../data/models/trace.dart';

class TraceTimeline extends StatelessWidget {
  final TraceData trace;

  const TraceTimeline({super.key, required this.trace});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Text('Execution Trace',
                    style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: Colors.grey,
                        letterSpacing: 1)),
                const Spacer(),
                TextButton.icon(
                  onPressed: () => context.push('/trace/${trace.traceId}'),
                  icon: const Icon(Icons.open_in_full, size: 14),
                  label: const Text('Full View', style: TextStyle(fontSize: 11)),
                  style: TextButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 8),
                    minimumSize: Size.zero,
                    tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            ...trace.steps.map((step) => _buildStepRow(step)),
          ],
        ),
      ),
    );
  }

  Widget _buildStepRow(TraceStep step) {
    final isSuccess = step.status == 'success';
    final isFailed = step.status == 'failed';
    final isSkipped = step.status == 'skipped';

    final icon = isSuccess
        ? Icons.check_circle_outline
        : isFailed
            ? Icons.error_outline
            : Icons.radio_button_unchecked;
    final color = isSuccess
        ? Colors.green
        : isFailed
            ? Colors.red
            : Colors.grey;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          Icon(icon, size: 14, color: color),
          const SizedBox(width: 8),
          SizedBox(
            width: 130,
            child: Text(step.type,
                style: TextStyle(color: color, fontSize: 12),
                overflow: TextOverflow.ellipsis),
          ),
          if (step.error != null && step.error!.isNotEmpty)
            Expanded(
              child: Text(step.error!,
                  style: const TextStyle(color: Colors.red, fontSize: 11),
                  overflow: TextOverflow.ellipsis),
            )
          else
            const Spacer(),
          Text(
            isSkipped ? 'skipped' : isFailed ? 'failed' : '',
            style: TextStyle(color: color, fontSize: 10),
          ),
        ],
      ),
    );
  }
}
