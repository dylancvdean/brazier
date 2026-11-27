import { useMemo } from 'react';
import { Linking, Platform, ScrollView, StyleSheet, Text, View } from 'react-native';
import { parseMarkdown, type Inline } from './markdown.ts';
import { useTheme } from './theme.ts';

const mono = Platform.select({ ios: 'Menlo', default: 'monospace' });

export function Markdown({ text }: { text: string }) {
  const p = useTheme();
  const blocks = useMemo(() => parseMarkdown(text), [text]);
  const styles = useMemo(() => StyleSheet.create({
    body: { fontSize: 16, lineHeight: 24, color: p.text },
    code: { fontFamily: mono, fontSize: 14, color: p.soft, backgroundColor: p.raised },
    link: { color: p.accent, textDecorationLine: 'underline' },
    pre: { borderRadius: 10, borderWidth: 1, borderColor: p.line, backgroundColor: p.panel },
    preText: { fontFamily: mono, fontSize: 13, lineHeight: 19, color: p.soft, padding: 11 },
    quote: { borderLeftWidth: 3, borderColor: p.line, paddingLeft: 11 },
    rule: { height: StyleSheet.hairlineWidth, backgroundColor: p.line, marginVertical: 4 },
    item: { flexDirection: 'row', gap: 8 },
    marker: { fontSize: 16, lineHeight: 24, color: p.muted, minWidth: 14 },
  }), [p]);

  const inline = (nodes: Inline[], prefix: string) => nodes.map((node, index) => {
    const key = `${prefix}.${index}`;
    if (node.type === 'text') return node.text;
    if (node.type === 'code') return <Text key={key} style={styles.code}>{node.text}</Text>;
    if (node.type === 'link') {
      return <Text key={key} style={styles.link} accessibilityRole="link"
        onPress={() => { void Linking.openURL(node.href).catch(() => undefined); }}>{inline(node.children, key)}</Text>;
    }
    return <Text key={key} style={node.type === 'strong' ? { fontWeight: '700' } : { fontStyle: 'italic' }}>
      {inline(node.children, key)}
    </Text>;
  });

  return <View style={{ gap: 9 }}>
    {blocks.map((block, index) => {
      const key = String(index);
      if (block.type === 'rule') return <View key={key} style={styles.rule} />;
      if (block.type === 'code') {
        return <ScrollView key={key} horizontal style={styles.pre}>
          <Text selectable style={styles.preText}>{block.text}</Text>
        </ScrollView>;
      }
      if (block.type === 'heading') {
        const size = [22, 20, 18, 17, 16, 16][block.level - 1];
        return <Text key={key} selectable accessibilityRole="header"
          style={[styles.body, { fontSize: size, lineHeight: size + 8, fontWeight: '700' }]}>
          {inline(block.children, key)}
        </Text>;
      }
      if (block.type === 'item') {
        return <View key={key} style={[styles.item, { marginLeft: block.depth * 16 }]}>
          <Text style={styles.marker}>{block.marker}</Text>
          <Text selectable style={[styles.body, { flex: 1 }]}>{inline(block.children, key)}</Text>
        </View>;
      }
      const body = <Text key={key} selectable style={[styles.body, block.type === 'quote' && { color: p.soft }]}>
        {inline(block.children, key)}
      </Text>;
      return block.type === 'quote' ? <View key={key} style={styles.quote}>{body}</View> : body;
    })}
  </View>;
}
