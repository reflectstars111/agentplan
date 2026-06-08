/// Upload dialog — text content upload with source name.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/backend_provider.dart';

class UploadDialog extends ConsumerStatefulWidget {
  const UploadDialog({super.key});

  @override
  ConsumerState<UploadDialog> createState() => _UploadDialogState();
}

class _UploadDialogState extends ConsumerState<UploadDialog> {
  final _contentController = TextEditingController();
  final _nameController = TextEditingController();
  bool _uploading = false;
  String? _result;

  @override
  void dispose() {
    _contentController.dispose();
    _nameController.dispose();
    super.dispose();
  }

  Future<void> _upload() async {
    final content = _contentController.text.trim();
    final name = _nameController.text.trim();
    if (content.isEmpty || name.isEmpty) return;

    setState(() {
      _uploading = true;
      _result = null;
    });

    try {
      final repo = ref.read(agentRepositoryProvider);
      final result = await repo.uploadText(content, name);
      setState(() {
        _uploading = false;
        _result = 'Uploaded: ${result.chunksCreated} chunks (${result.sourceId})';
      });
    } catch (e) {
      setState(() {
        _uploading = false;
        _result = 'Upload failed: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Upload Text Knowledge'),
      content: SizedBox(
        width: 500,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: _nameController,
              decoration: const InputDecoration(
                labelText: 'Source Name',
                hintText: 'e.g., research_paper.txt, notes.md',
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _contentController,
              maxLines: 8,
              decoration: const InputDecoration(
                labelText: 'Content',
                hintText: 'Paste your text content here...',
                alignLabelWithHint: true,
              ),
            ),
            if (_result != null) ...[
              const SizedBox(height: 12),
              Text(_result!,
                  style: TextStyle(
                      color: _result!.contains('failed')
                          ? Colors.red
                          : Colors.green,
                      fontSize: 12)),
            ],
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed: _uploading ? null : _upload,
          child: Text(_uploading ? 'Uploading...' : 'Upload'),
        ),
      ],
    );
  }
}
