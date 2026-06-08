/// Dio-based HTTP client for the Agent-OS backend REST API.
///
/// Provides typed methods for all API endpoints with error handling.
library;

import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'backend_process.dart';
import '../data/models/query_response.dart';
import '../data/models/task_response.dart';
import '../data/models/trace.dart';
import '../data/models/upload_response.dart';

/// Encapsulates all Agent-OS API calls.
class AgentApiClient {
  final Dio _dio;

  AgentApiClient({required String baseUrl})
      : _dio = Dio(BaseOptions(
          baseUrl: baseUrl,
          connectTimeout: const Duration(seconds: 10),
          receiveTimeout: const Duration(seconds: 60),
          headers: {'Content-Type': 'application/json'},
        ));

  /// Check if the backend is healthy.
  Future<bool> healthCheck() async {
    try {
      final response = await _dio.get('/health');
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Upload text content for indexing.
  Future<UploadResponse> uploadText(String content, String sourceName) async {
    final response = await _dio.post('/upload', data: {
      'content': content,
      'source_name': sourceName,
    });
    return UploadResponse.fromJson(response.data);
  }

  /// Upload a file for indexing.
  Future<UploadResponse> uploadFile(String filePath, String fileName) async {
    final formData = FormData.fromMap({
      'file': await MultipartFile.fromFile(filePath, filename: fileName),
    });
    final response = await _dio.post('/upload/file', data: formData);
    return UploadResponse.fromJson(response.data);
  }

  /// Clone and index a GitHub repository.
  Future<Map<String, dynamic>> uploadGithub(
    String repoUrl, {
    String branch = 'main',
  }) async {
    final response = await _dio.post('/upload/github', data: {
      'repo_url': repoUrl,
      'branch': branch,
    });
    return response.data;
  }

  /// Fetch and index a web URL.
  Future<Map<String, dynamic>> uploadUrl(String url, {String? sourceName}) async {
    final response = await _dio.post('/upload/url', data: {
      'url': url,
      if (sourceName != null) 'source_name': sourceName,
    });
    return response.data;
  }

  /// Execute a simple-mode query (AgentRuntime.process_query).
  Future<QueryResponse> querySimple(String query, {String? model}) async {
    final response = await _dio.post('/query', data: {
      'query': query,
      if (model != null) 'model': model,
    });
    return QueryResponse.fromJson(response.data);
  }

  /// Execute a task-mode query (Controller.process).
  Future<TaskResponse> queryTask(String query, {String? model}) async {
    final response = await _dio.post('/task', data: {
      'query': query,
      if (model != null) 'model': model,
    });
    return TaskResponse.fromJson(response.data);
  }

  /// Retrieve an execution trace by ID.
  Future<TraceData> getTrace(String traceId) async {
    final response = await _dio.get('/trace/$traceId');
    if (response.statusCode != 200) {
      throw DioException(
        requestOptions: response.requestOptions,
        message: 'Trace not found',
      );
    }
    return TraceData.fromJson(response.data);
  }

  /// Write output to a file.
  Future<Map<String, dynamic>> writeFile(
    String content,
    String filename, {
    String format = 'text',
  }) async {
    final response = await _dio.post('/output/file', data: {
      'content': content,
      'filename': filename,
      'format': format,
    });
    return response.data;
  }

  void dispose() {
    _dio.close();
  }
}

/// Riverpod provider for the API client.
final apiClientProvider = Provider<AgentApiClient>((ref) {
  final backend = ref.watch(backendProcessProvider);
  return AgentApiClient(baseUrl: backend.baseUrl);
});
