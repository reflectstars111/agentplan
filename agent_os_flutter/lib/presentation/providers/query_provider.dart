/// Riverpod providers for query execution and response state.
library;

import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../../data/models/trace.dart';
import 'backend_provider.dart';

/// Current query mode: simple or task.
final queryModeProvider = StateProvider<String>((ref) => 'simple');

/// Selected model name.
final selectedModelProvider = StateProvider<String>((ref) => 'deepseek-chat');

/// Current conversation ID (null = new conversation).
final activeConversationIdProvider = StateProvider<int?>((ref) => null);

/// Query execution state.
enum QueryState { idle, loading, success, error }

/// Query execution state notifier.
class QueryNotifier extends StateNotifier<QueryState> {
  final Ref _ref;
  String _response = '';
  String _traceId = '';
  bool _verified = false;
  Map<String, dynamic>? _intent;
  Map<String, dynamic>? _taskGraph;
  TraceData? _traceData;
  List<String> _conflicts = [];
  List<String> _suggestions = [];
  Map<String, dynamic> _writeback = {};
  String _errorMessage = '';

  QueryNotifier(this._ref) : super(QueryState.idle);

  String get response => _response;
  String get traceId => _traceId;
  bool get verified => _verified;
  Map<String, dynamic>? get intent => _intent;
  Map<String, dynamic>? get taskGraph => _taskGraph;
  TraceData? get traceData => _traceData;
  List<String> get conflicts => _conflicts;
  List<String> get suggestions => _suggestions;
  Map<String, dynamic> get writeback => _writeback;
  String get errorMessage => _errorMessage;

  Future<void> executeQuery(String query) async {
    if (query.trim().isEmpty) return;

    state = QueryState.loading;
    _response = '';
    _traceId = '';
    _traceData = null;
    _intent = null;
    _taskGraph = null;
    _errorMessage = '';

    try {
      final repo = _ref.read(agentRepositoryProvider);
      final mode = _ref.read(queryModeProvider);
      final model = _ref.read(selectedModelProvider);

      if (mode == 'task') {
        final result = await repo.queryTask(query, model: model);
        _response = result.response;
        _traceId = result.traceId;
        _intent = result.intent;
        _taskGraph = result.taskGraphSummary;
        _conflicts = result.suggestions;
      } else {
        final result = await repo.querySimple(query, model: model);
        _response = result.response;
        _traceId = result.traceId;
        _verified = result.verified;
        _conflicts = result.conflictingPairs
            .map((p) => p.join(' vs '))
            .toList();
        _suggestions = result.suggestions;
        _writeback = result.writeback;
      }

      // Fetch trace if available
      if (_traceId.isNotEmpty) {
        try {
          _traceData = await repo.getTrace(_traceId);
        } catch (_) {
          // Trace may not be available yet
        }
      }

      state = QueryState.success;
    } catch (e) {
      _errorMessage = e.toString();
      _response = 'Error: $_errorMessage';
      state = QueryState.error;
    }
  }

  void reset() {
    state = QueryState.idle;
    _response = '';
    _traceId = '';
    _verified = false;
    _traceData = null;
    _intent = null;
    _taskGraph = null;
  }
}

/// Provider for the query notifier.
final queryNotifierProvider =
    StateNotifierProvider<QueryNotifier, QueryState>((ref) {
  return QueryNotifier(ref);
});
