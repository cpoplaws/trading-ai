import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import {
  Card,
  ErrorState,
  formatCurrency,
  Header,
  IconButton,
  LoadingState,
  Metric,
  QuickLink,
  Screen,
  SectionHeading,
  StatusPill,
  styles as uiStyles,
} from '@/components/TradingUI';
import { usePortfolioHistoryQuery, usePortfolioQuery } from '@/lib/trading-api';
import { useColors } from '@/hooks/useColors';

function PortfolioCard() {
  const colors = useColors();
  const { data, isLoading, isError, error, refetch } = usePortfolioQuery();
  const { data: historyResponse } = usePortfolioHistoryQuery();
  const history = Array.isArray(historyResponse) ? historyResponse : historyResponse?.history ?? [];
  const historyValues = history.map((point) => point.value ?? point.total_value ?? 0).filter((value) => value > 0);
  const pnl = data?.daily_pnl ?? 0;
  const chartMax = Math.max(...historyValues, 1);

  if (isLoading) return <Card><LoadingState label="Loading portfolio…" /></Card>;
  if (isError) return <ErrorState message={error instanceof Error ? error.message : 'The portfolio endpoint is unavailable.'} onRetry={() => void refetch()} />;

  return (
    <Card style={styles.portfolioCard}>
      <View style={styles.portfolioTop}>
        <View>
          <Text style={[uiStyles.metricLabel, { color: colors.mutedForeground }]}>TOTAL PORTFOLIO</Text>
          <Text style={[styles.portfolioValue, { color: colors.foreground }]}>{formatCurrency(data?.total_value)}</Text>
          <View style={styles.pnlRow}>
            <Ionicons name={pnl >= 0 ? 'trending-up' : 'trending-down'} size={14} color={pnl >= 0 ? colors.success : colors.destructive} />
            <Text style={[styles.pnlText, { color: pnl >= 0 ? colors.success : colors.destructive }]}>
              {pnl >= 0 ? '+' : ''}{formatCurrency(pnl)} ({(data?.daily_pnl_percent ?? 0).toFixed(2)}%)
            </Text>
            <Text style={[styles.todayText, { color: colors.mutedForeground }]}>today</Text>
          </View>
        </View>
        <StatusPill active label={data?.demo_mode ? 'DEMO MODE' : 'LIVE'} />
      </View>
      <View style={styles.chart}>
        {historyValues.length > 1 ? historyValues.map((value, index) => (
          <View key={`${value}-${index}`} style={styles.chartBarWrap}>
            <View style={[styles.chartBar, { height: Math.max(10, (value / chartMax) * 66), backgroundColor: index === historyValues.length - 1 ? colors.primary : colors.accent }]} />
          </View>
        )) : <Text style={[styles.chartEmpty, { color: colors.mutedForeground }]}>Portfolio history will appear after your first session</Text>}
      </View>
      <View style={[styles.portfolioFooter, { borderTopColor: colors.border }]}>
        <View><Text style={[styles.footerLabel, { color: colors.mutedForeground }]}>Positions</Text><Text style={[styles.footerValue, { color: colors.foreground }]}>{data?.positions_count ?? 0}</Text></View>
        <View><Text style={[styles.footerLabel, { color: colors.mutedForeground }]}>Mode</Text><Text style={[styles.footerValue, { color: colors.foreground }]}>{data?.demo_mode ? 'Paper' : 'Alpaca'}</Text></View>
        <Ionicons name="arrow-up-outline" size={18} color={colors.mutedForeground} />
      </View>
    </Card>
  );
}

export default function OverviewScreen() {
  const colors = useColors();
  const router = useRouter();
  const { data, isRefetching, refetch } = usePortfolioQuery();
  const portfolio = data;

  return (
    <Screen refreshing={isRefetching} onRefresh={() => void refetch()}>
      <Header
        eyebrow="QUANTLYTICS / CONTROL ROOM"
        title="Good morning."
        subtitle="Your autonomous trading stack at a glance."
        action={<IconButton icon="notifications-outline" label="Notifications" />}
      />
      <PortfolioCard />
      <View style={styles.sectionGap} />
      <SectionHeading title="Performance snapshot" action="Refresh" onAction={() => void refetch()} />
      <View style={styles.metricsGrid}>
        <Metric label="Cash available" value={formatCurrency(portfolio?.cash, true)} hint={`Buying power ${formatCurrency(portfolio?.buying_power, true)}`} />
        <Metric label="Sharpe ratio" value={(portfolio?.sharpe_ratio ?? 0).toFixed(2)} hint="Risk-adjusted return" />
        <Metric label="Win rate" value={`${((portfolio?.win_rate ?? 0) * 100).toFixed(1)}%`} hint="Successful trades" positive={(portfolio?.win_rate ?? 0) > 0.5} />
        <Metric label="Daily P&L" value={formatCurrency(portfolio?.daily_pnl)} hint={`${(portfolio?.daily_pnl_percent ?? 0).toFixed(2)}% today`} positive={(portfolio?.daily_pnl ?? 0) >= 0} />
      </View>
      <View style={styles.sectionGap} />
      <SectionHeading title="Control room" />
      <View style={styles.quickGrid}>
        <QuickLink icon="layers-outline" label="Strategies" value="Tune your edge" color={colors.accent} onPress={() => router.push('/(tabs)/strategies' as never)} />
        <QuickLink icon="git-network-outline" label="Agent swarm" value="Coordinate AI" color={colors.secondary} onPress={() => router.push('/(tabs)/agents' as never)} />
        <QuickLink icon="swap-vertical-outline" label="Recent trades" value="Review activity" color={colors.secondary} onPress={() => router.push('/(tabs)/trades' as never)} />
      </View>
      <View style={styles.sectionGap} />
      <Card style={styles.signalCard}>
        <View style={[styles.signalIcon, { backgroundColor: colors.accent }]}><Ionicons name="pulse-outline" size={20} color={colors.primary} /></View>
        <View style={styles.signalCopy}>
          <Text style={[styles.signalTitle, { color: colors.foreground }]}>System health</Text>
          <Text style={[styles.signalText, { color: colors.mutedForeground }]}>Live data polling is active. Use pull to refresh anytime.</Text>
        </View>
        <StatusPill active label="ONLINE" />
      </Card>
    </Screen>
  );
}

const styles = StyleSheet.create({
  portfolioCard: { padding: 18, overflow: 'hidden' },
  portfolioTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  portfolioValue: { fontFamily: 'Inter_700Bold', fontSize: 35, letterSpacing: -1.1, marginTop: 8 },
  pnlRow: { flexDirection: 'row', alignItems: 'center', gap: 5, marginTop: 7 },
  pnlText: { fontFamily: 'Inter_600SemiBold', fontSize: 12 },
  todayText: { fontFamily: 'Inter_400Regular', fontSize: 12, marginLeft: 2 },
  chart: { height: 88, marginTop: 20, flexDirection: 'row', alignItems: 'flex-end', gap: 5 },
  chartBarWrap: { flex: 1, justifyContent: 'flex-end', height: 70 },
  chartBar: { width: '100%', borderRadius: 5, minHeight: 5 },
  chartEmpty: { fontFamily: 'Inter_400Regular', fontSize: 11, paddingBottom: 12 },
  portfolioFooter: { borderTopWidth: 1, paddingTop: 14, marginTop: 13, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  footerLabel: { fontFamily: 'Inter_400Regular', fontSize: 10 },
  footerValue: { fontFamily: 'Inter_600SemiBold', fontSize: 13, marginTop: 4 },
  sectionGap: { height: 26 },
  metricsGrid: { paddingHorizontal: 20, flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between' },
  quickGrid: { paddingHorizontal: 20, flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  signalCard: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  signalIcon: { width: 38, height: 38, borderRadius: 13, alignItems: 'center', justifyContent: 'center' },
  signalCopy: { flex: 1 },
  signalTitle: { fontFamily: 'Inter_600SemiBold', fontSize: 13 },
  signalText: { fontFamily: 'Inter_400Regular', fontSize: 11, lineHeight: 16, marginTop: 3 },
});
