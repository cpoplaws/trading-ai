import { Feather, Ionicons } from '@expo/vector-icons';
import { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { AGENT_INFO, Card, ErrorState, Header, LoadingState, Screen, SectionHeading, StatusPill, Toggle } from '@/components/TradingUI';
import { useAgentToggleMutation, useDecisionsQuery, useSwarmQuery, useSwarmToggleMutation } from '@/lib/trading-api';
import { useColors } from '@/hooks/useColors';

export default function AgentsScreen() {
  const colors = useColors();
  const swarm = useSwarmQuery();
  const decisions = useDecisionsQuery();
  const swarmMutation = useSwarmToggleMutation();
  const agentMutation = useAgentToggleMutation();
  const [pending, setPending] = useState<string | null>(null);

  const toggleSwarm = async () => {
    if (!swarm.data) return;
    setPending('swarm');
    try { await swarmMutation.mutateAsync(!swarm.data.enabled); } finally { setPending(null); }
  };
  const toggleAgent = async (name: string) => {
    setPending(name);
    try { await agentMutation.mutateAsync(name); } finally { setPending(null); }
  };

  return (
    <Screen refreshing={swarm.isRefetching || decisions.isRefetching} onRefresh={() => { void swarm.refetch(); void decisions.refetch(); }}>
      <Header eyebrow="QUANTLYTICS / INTELLIGENCE" title="Agent swarm" subtitle="Coordinate specialized agents while you keep the final say." action={<StatusPill active={swarm.data?.enabled ?? false} label={swarm.data?.enabled ? 'RUNNING' : 'PAUSED'} />} />
      {swarm.isLoading ? <Card><LoadingState label="Connecting to agent swarm…" /></Card> : swarm.isError ? <ErrorState message={swarm.error instanceof Error ? swarm.error.message : 'Agent status is unavailable.'} onRetry={() => void swarm.refetch()} /> : (
        <>
          <Card style={styles.swarmCard}>
            <View style={[styles.swarmIcon, { backgroundColor: colors.accent }]}><Ionicons name="git-network-outline" size={21} color={colors.primary} /></View>
            <View style={styles.swarmCopy}>
              <Text style={[styles.swarmTitle, { color: colors.foreground }]}>Swarm coordination</Text>
              <Text style={[styles.swarmSubtitle, { color: colors.mutedForeground }]}>{swarm.data?.coordination_mode ?? 'Autonomous coordination'}</Text>
            </View>
            <Toggle enabled={swarm.data?.enabled ?? false} loading={pending === 'swarm'} label="Agent swarm" testID="swarm-toggle" onPress={() => void toggleSwarm()} />
          </Card>
          <View style={styles.sectionGap} />
          <SectionHeading title="Specialists" action={`${Object.values(swarm.data?.agents ?? {}).filter((agent) => agent.enabled).length ?? 0} online`} />
          <View style={styles.agentList}>
            {Object.entries(swarm.data?.agents ?? {}).map(([key, agent]) => {
              const info = AGENT_INFO[key] ?? { name: key, description: 'Specialized trading agent', icon: 'cpu' as const, color: 'blue' as const };
              const accent = info.color === 'green' ? colors.success : info.color === 'orange' ? colors.warning : info.color === 'violet' ? colors.accentForeground : colors.primary;
              return (
                <Card key={key} style={styles.agentCard}>
                  <View style={styles.agentTop}>
                    <View style={[styles.agentIcon, { backgroundColor: agent.enabled ? colors.accent : colors.secondary }]}><Feather name={info.icon} size={17} color={agent.enabled ? accent : colors.mutedForeground} /></View>
                    <Toggle enabled={agent.enabled} loading={pending === key} label={`${info.name} agent`} testID={`agent-toggle-${key}`} onPress={() => void toggleAgent(key)} />
                  </View>
                  <Text style={[styles.agentName, { color: colors.foreground }]}>{info.name}</Text>
                  <Text style={[styles.agentDescription, { color: colors.mutedForeground }]}>{info.description}</Text>
                  <View style={styles.agentStats}><View><Text style={[styles.statLabel, { color: colors.mutedForeground }]}>Accuracy</Text><Text style={[styles.statValue, { color: colors.foreground }]}>{(agent.performance.accuracy * 100).toFixed(1)}%</Text></View><View><Text style={[styles.statLabel, { color: colors.mutedForeground }]}>Decisions</Text><Text style={[styles.statValue, { color: colors.foreground }]}>{agent.performance.total_decisions}</Text></View></View>
                </Card>
              );
            })}
          </View>
          <View style={styles.sectionGap} />
          <SectionHeading title="Latest decisions" />
          {decisions.isError ? <Text style={[styles.inlineError, { color: colors.mutedForeground }]}>Decision stream is unavailable.</Text> : decisions.data?.length ? decisions.data.map((decision, index) => (
            <View key={`${decision.timestamp}-${index}`} style={[styles.decision, { borderBottomColor: colors.border }]}>
              <View style={[styles.decisionDot, { backgroundColor: decision.action.toLowerCase().includes('sell') ? colors.destructive : colors.success }]} />
              <View style={styles.decisionCopy}><Text style={[styles.decisionTitle, { color: colors.foreground }]}>{decision.agent} · {decision.action}</Text><Text style={[styles.decisionReason, { color: colors.mutedForeground }]} numberOfLines={1}>{decision.reason}</Text></View><Text style={[styles.confidence, { color: colors.primary }]}>{(decision.confidence * 100).toFixed(0)}%</Text>
            </View>
          )) : <Text style={[styles.inlineError, { color: colors.mutedForeground }]}>No recent decisions from the swarm.</Text>}
        </>
      )}
    </Screen>
  );
}

const styles = StyleSheet.create({
  swarmCard: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  swarmIcon: { width: 40, height: 40, borderRadius: 14, alignItems: 'center', justifyContent: 'center' },
  swarmCopy: { flex: 1 },
  swarmTitle: { fontFamily: 'Inter_600SemiBold', fontSize: 14 },
  swarmSubtitle: { fontFamily: 'Inter_400Regular', fontSize: 11, marginTop: 4 },
  sectionGap: { height: 26 },
  agentList: { paddingHorizontal: 20, gap: 12 },
  agentCard: { marginHorizontal: 0, padding: 15 },
  agentTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  agentIcon: { width: 34, height: 34, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
  agentName: { fontFamily: 'Inter_600SemiBold', fontSize: 14, marginTop: 13 },
  agentDescription: { fontFamily: 'Inter_400Regular', fontSize: 11, marginTop: 4 },
  agentStats: { flexDirection: 'row', gap: 30, marginTop: 16 },
  statLabel: { fontFamily: 'Inter_400Regular', fontSize: 10 },
  statValue: { fontFamily: 'Inter_700Bold', fontSize: 15, marginTop: 4 },
  decision: { marginHorizontal: 20, paddingVertical: 13, borderBottomWidth: 1, flexDirection: 'row', alignItems: 'center', gap: 10 },
  decisionDot: { width: 7, height: 7, borderRadius: 4 },
  decisionCopy: { flex: 1 },
  decisionTitle: { fontFamily: 'Inter_600SemiBold', fontSize: 12 },
  decisionReason: { fontFamily: 'Inter_400Regular', fontSize: 10, marginTop: 4 },
  confidence: { fontFamily: 'Inter_600SemiBold', fontSize: 11 },
  inlineError: { paddingHorizontal: 20, fontFamily: 'Inter_400Regular', fontSize: 12, paddingVertical: 12 },
});