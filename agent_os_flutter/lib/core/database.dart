/// Local SQLite database using drift.
///
/// Stores conversation history, messages, and settings for persistence
/// across app restarts.
library;

import 'dart:io';
import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

part 'database.g.dart';

// ── Tables ──────────────────────────────────────────────────────

/// Conversation sessions (one per topic/thread).
class Conversations extends Table {
  IntColumn get id => integer().autoIncrement()();
  TextColumn get title => text()();
  DateTimeColumn get createdAt => dateTime().withDefault(currentDateAndTime)();
  DateTimeColumn get updatedAt => dateTime().withDefault(currentDateAndTime)();
  TextColumn get mode => text().withDefault(const Constant('simple'))();
}

/// Messages within a conversation (user queries + agent responses).
class Messages extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get conversationId => integer().references(Conversations, #id)();
  TextColumn get role => text()();       // 'user' | 'agent'
  TextColumn get content => text()();
  TextColumn get model => text().nullable()();
  TextColumn get traceId => text().nullable()();
  BoolColumn get verified => boolean().nullable()();
  DateTimeColumn get createdAt => dateTime().withDefault(currentDateAndTime)();
}

/// Application settings (key-value store).
class Settings extends Table {
  TextColumn get key => text()();
  TextColumn get value => text()();
  DateTimeColumn get updatedAt => dateTime().withDefault(currentDateAndTime)();

  @override
  Set<Column> get primaryKey => {key};
}

// ── Database ─────────────────────────────────────────────────────

@DriftDatabase(tables: [Conversations, Messages, Settings])
class AppDatabase extends _$AppDatabase {
  AppDatabase() : super(_openConnection());

  @override
  int get schemaVersion => 1;

  @override
  MigrationStrategy get migration => MigrationStrategy(
    onCreate: (m) async {
      await m.createAll();
      // Seed default settings
      await into(settings).insertOnConflictUpdate(SettingsCompanion(
        key: const Value('backend_host'),
        value: const Value('127.0.0.1'),
      ));
      await into(settings).insertOnConflictUpdate(SettingsCompanion(
        key: const Value('backend_port'),
        value: const Value('8000'),
      ));
      await into(settings).insertOnConflictUpdate(SettingsCompanion(
        key: const Value('default_model'),
        value: const Value('deepseek-chat'),
      ));
      await into(settings).insertOnConflictUpdate(SettingsCompanion(
        key: const Value('theme_mode'),
        value: const Value('dark'),
      ));
    },
  );

  /// Ensure the database file and tables exist.
  Future<void> initialize() async {
    // Accessing a table triggers lazy initialization
    await select(conversations).get();
    await select(messages).get();
    await select(settings).get();
  }

  // ── Conversation queries ───────────────────────────────────────

  /// List conversations ordered by most recent first.
  Future<List<Conversation>> listConversations({int limit = 50}) {
    return (select(conversations)
      ..orderBy([(t) => OrderingTerm.desc(t.updatedAt)])
      ..limit(limit))
    .get();
  }

  /// Create a new conversation and return its ID.
  Future<int> createConversation(String title, {String mode = 'simple'}) {
    return into(conversations).insert(ConversationsCompanion(
      title: Value(title),
      mode: Value(mode),
    ));
  }

  /// Touch a conversation (update its updatedAt).
  Future<void> touchConversation(int id) {
    return (update(conversations)..where((t) => t.id.equals(id)))
        .write(ConversationsCompanion(updatedAt: Value(DateTime.now())));
  }

  /// Delete a conversation and its messages.
  Future<void> deleteConversation(int id) {
    return transaction(() async {
      await (delete(messages)..where((t) => t.conversationId.equals(id))).go();
      await (delete(conversations)..where((t) => t.id.equals(id))).go();
    });
  }

  // ── Message queries ────────────────────────────────────────────

  /// Add a message to a conversation.
  Future<int> addMessage(
    int conversationId,
    String role,
    String content, {
    String? model,
    String? traceId,
    bool? verified,
  }) {
    return into(messages).insert(MessagesCompanion(
      conversationId: Value(conversationId),
      role: Value(role),
      content: Value(content),
      model: Value(model),
      traceId: Value(traceId),
      verified: Value(verified),
    ));
  }

  /// Get all messages for a conversation.
  Future<List<Message>> getMessages(int conversationId) {
    return (select(messages)
      ..where((t) => t.conversationId.equals(conversationId))
      ..orderBy([(t) => OrderingTerm.asc(t.createdAt)]))
    .get();
  }

  // ── Settings queries ───────────────────────────────────────────

  /// Get a setting value by key.
  Future<String?> getSetting(String key) async {
    final row = await (select(settings)
      ..where((t) => t.key.equals(key)))
    .getSingleOrNull();
    return row?.value;
  }

  /// Set a setting value.
  Future<void> setSetting(String key, String value) {
    return into(settings).insertOnConflictUpdate(SettingsCompanion(
      key: Value(key),
      value: Value(value),
      updatedAt: Value(DateTime.now()),
    ));
  }

  // ── Helpers ────────────────────────────────────────────────────

  static QueryExecutor _openConnection() {
    return LazyDatabase(() async {
      final dbFolder = await getApplicationDocumentsDirectory();
      final file = File(p.join(dbFolder.path, 'agent_os.db'));
      return NativeDatabase.createInBackground(file);
    });
  }
}

// ── Companion extensions (for insert helpers) ───────────────────

extension ConversationExtensions on ConversationsCompanion {
  static ConversationsCompanion insert({
    required String title,
    String? mode,
  }) {
    return ConversationsCompanion(
      title: Value(title),
      mode: Value(mode ?? 'simple'),
    );
  }
}

extension MessageExtensions on MessagesCompanion {
  static MessagesCompanion insert({
    required int conversationId,
    required String role,
    required String content,
    String? model,
    String? traceId,
    bool? verified,
  }) {
    return MessagesCompanion(
      conversationId: Value(conversationId),
      role: Value(role),
      content: Value(content),
      model: Value(model),
      traceId: Value(traceId),
      verified: Value(verified),
    );
  }
}

extension SettingExtensions on SettingsCompanion {
  static SettingsCompanion insert({
    required String key,
    required String value,
  }) {
    return SettingsCompanion(
      key: Value(key),
      value: Value(value),
    );
  }
}

/// Riverpod provider for the app database singleton.
final appDatabaseProvider = Provider<AppDatabase>((ref) {
  throw UnimplementedError('Must be overridden in main()');
});
