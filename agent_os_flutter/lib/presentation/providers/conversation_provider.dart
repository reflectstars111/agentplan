/// Riverpod provider for conversation history management.
library;

import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../core/database.dart';
import 'backend_provider.dart';

/// List of all conversations.
final conversationsProvider = FutureProvider<List<Conversation>>((ref) async {
  final repo = ref.watch(localRepositoryProvider);
  return repo.listConversations();
});

/// Messages for the active conversation.
final activeMessagesProvider = FutureProvider.family<List<Message>, int>((ref, conversationId) async {
  final repo = ref.watch(localRepositoryProvider);
  return repo.getMessages(conversationId);
});

/// Notifier for conversation CRUD operations.
class ConversationNotifier extends StateNotifier<AsyncValue<void>> {
  final Ref _ref;
  ConversationNotifier(this._ref) : super(const AsyncValue.data(null));

  Future<int> createConversation(String title, {String mode = 'simple'}) async {
    final repo = _ref.read(localRepositoryProvider);
    final id = await repo.createConversation(title, mode: mode);
    _ref.invalidate(conversationsProvider);
    return id;
  }

  Future<void> deleteConversation(int id) async {
    final repo = _ref.read(localRepositoryProvider);
    await repo.deleteConversation(id);
    _ref.invalidate(conversationsProvider);
  }

  Future<void> addMessage(
    int conversationId,
    String role,
    String content, {
    String? model,
    String? traceId,
    bool? verified,
  }) async {
    final repo = _ref.read(localRepositoryProvider);
    await repo.addMessage(
      conversationId,
      role,
      content,
      model: model,
      traceId: traceId,
      verified: verified,
    );
    await repo.touchConversation(conversationId);
    _ref.invalidate(conversationsProvider);
    _ref.invalidate(activeMessagesProvider(conversationId));
  }
}

final conversationNotifierProvider =
    StateNotifierProvider<ConversationNotifier, AsyncValue<void>>((ref) {
  return ConversationNotifier(ref);
});
