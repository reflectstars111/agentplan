/// Sidebar — upload buttons + conversation history list.
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:file_picker/file_picker.dart';
import '../../core/database.dart';
import '../providers/backend_provider.dart';
import '../providers/conversation_provider.dart';
import 'upload_dialog.dart';

class Sidebar extends ConsumerWidget {
  final VoidCallback onNewConversation;
  final void Function(int id) onSelectConversation;
  final int? currentConversationId;

  const Sidebar({
    super.key,
    required this.onNewConversation,
    required this.onSelectConversation,
    this.currentConversationId,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final conversations = ref.watch(conversationsProvider).valueOrNull ?? [];

    return SizedBox(
      width: 240,
      child: Column(
        children: [
          // New conversation button
          Padding(
            padding: const EdgeInsets.all(8),
            child: FilledButton.icon(
              onPressed: onNewConversation,
              icon: const Icon(Icons.add, size: 18),
              label: const Text('New Query'),
              style: FilledButton.styleFrom(
                minimumSize: const Size(double.infinity, 40),
              ),
            ),
          ),

          // Upload section
          _UploadSection(),

          const Divider(),

          // Conversation list header
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: Row(
              children: [
                const Text('HISTORY',
                    style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                        color: Colors.grey,
                        letterSpacing: 1.5)),
                const Spacer(),
                if (conversations.isNotEmpty)
                  Text('${conversations.length}',
                      style: const TextStyle(
                          color: Colors.grey, fontSize: 10)),
              ],
            ),
          ),

          // Conversation list
          Expanded(
            child: conversations.isEmpty
                ? const Center(
                    child: Text('No conversations yet',
                        style: TextStyle(color: Colors.grey, fontSize: 12)),
                  )
                : ListView.builder(
                    itemCount: conversations.length,
                    itemBuilder: (context, index) {
                      final conv = conversations[index];
                      final isActive = conv.id == currentConversationId;
                      return _ConversationTile(
                        conversation: conv,
                        isActive: isActive,
                        onTap: () => onSelectConversation(conv.id),
                        onDelete: () => ref
                            .read(conversationNotifierProvider.notifier)
                            .deleteConversation(conv.id),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}

class _UploadSection extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8),
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Padding(
                padding: EdgeInsets.only(left: 4, bottom: 4),
                child: Text('UPLOAD',
                    style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                        color: Colors.grey,
                        letterSpacing: 1.5)),
              ),
              _uploadButton(
                icon: Icons.text_fields,
                label: 'Text',
                onTap: () => _showUploadDialog(context, ref),
              ),
              _uploadButton(
                icon: Icons.attach_file,
                label: 'File',
                onTap: () => _pickAndUploadFile(context, ref),
              ),
              _uploadButton(
                icon: Icons.code,
                label: 'GitHub',
                onTap: () => _showGitHubDialog(context, ref),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _uploadButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
  }) {
    return SizedBox(
      width: double.infinity,
      height: 32,
      child: TextButton.icon(
        onPressed: onTap,
        icon: Icon(icon, size: 16),
        label: Text(label, style: const TextStyle(fontSize: 12)),
        style: TextButton.styleFrom(
          padding: const EdgeInsets.symmetric(horizontal: 8),
          alignment: Alignment.centerLeft,
        ),
      ),
    );
  }

  void _showUploadDialog(BuildContext context, WidgetRef ref) {
    showDialog(
      context: context,
      builder: (_) => const UploadDialog(),
    );
  }

  Future<void> _pickAndUploadFile(BuildContext context, WidgetRef ref) async {
    final result = await FilePicker.platform.pickFiles();
    if (result == null || result.files.isEmpty) return;

    final file = result.files.first;
    if (file.path == null) return;

    try {
      final repo = ref.read(agentRepositoryProvider);
      await repo.uploadFile(file.path!, file.name);
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Uploaded: ${file.name}')),
        );
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Upload failed: $e')),
        );
      }
    }
  }

  void _showGitHubDialog(BuildContext context, WidgetRef ref) {
    final urlController = TextEditingController();
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Clone GitHub Repo'),
        content: TextField(
          controller: urlController,
          decoration: const InputDecoration(
            hintText: 'https://github.com/user/repo',
            labelText: 'Repository URL',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () async {
              Navigator.pop(ctx);
              final url = urlController.text.trim();
              if (url.isEmpty) return;
              try {
                final repo = ref.read(agentRepositoryProvider);
                await repo.uploadGithub(url);
                if (context.mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Cloning $url...')),
                  );
                }
              } catch (e) {
                if (context.mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('GitHub upload failed: $e')),
                  );
                }
              }
            },
            child: const Text('Clone'),
          ),
        ],
      ),
    );
  }
}

class _ConversationTile extends StatelessWidget {
  final Conversation conversation;
  final bool isActive;
  final VoidCallback onTap;
  final VoidCallback onDelete;

  const _ConversationTile({
    required this.conversation,
    required this.isActive,
    required this.onTap,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    return ListTile(
      selected: isActive,
      selectedTileColor: Colors.cyan.withAlpha(15),
      dense: true,
      title: Text(
        conversation.title,
        maxLines: 1,
        overflow: TextOverflow.ellipsis,
        style: TextStyle(
          fontSize: 12,
          color: isActive ? Colors.cyan : null,
          fontWeight: isActive ? FontWeight.w600 : FontWeight.normal,
        ),
      ),
      subtitle: Text(
        _formatDate(conversation.updatedAt),
        style: const TextStyle(fontSize: 10, color: Colors.grey),
      ),
      trailing: IconButton(
        icon: const Icon(Icons.close, size: 14, color: Colors.grey),
        onPressed: onDelete,
        padding: EdgeInsets.zero,
        constraints: const BoxConstraints(),
      ),
      onTap: onTap,
    );
  }

  String _formatDate(DateTime date) {
    final now = DateTime.now();
    final diff = now.difference(date);
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    if (diff.inDays < 7) return '${diff.inDays}d ago';
    return '${date.month}/${date.day}/${date.year}';
  }
}
