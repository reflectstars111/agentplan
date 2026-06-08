import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:agent_os_flutter/app.dart';

void main() {
  testWidgets('AgentOSApp can be created', (WidgetTester tester) async {
    await tester.pumpWidget(
      const ProviderScope(child: AgentOSApp()),
    );
    // The app should render without errors
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
