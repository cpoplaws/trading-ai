/**
 * Trading AI Mobile App
 * React Native / Expo Application
 */
import React, { useEffect, useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons } from '@expo/vector-icons';
import { StatusBar } from 'expo-status-bar';
import * as SecureStore from 'expo-secure-store';

// Screens
import DashboardScreen from './screens/DashboardScreen';
import PortfolioScreen from './screens/PortfolioScreen';
import SignalsScreen from './screens/SignalsScreen';
import SettingsScreen from './screens/SettingsScreen';
import LoginScreen from './screens/LoginScreen';

// API Client
import { apiClient } from './lib/api-client';

const Tab = createBottomTabNavigator();

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    checkAuthentication();
  }, []);

  const checkAuthentication = async () => {
    try {
      const apiKey = await SecureStore.getItemAsync('api_key');
      if (apiKey) {
        apiClient.setApiKey(apiKey);
        setIsAuthenticated(true);
      }
    } catch (error) {
      console.error('Failed to load API key:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogin = async (apiKey: string) => {
    try {
      await SecureStore.setItemAsync('api_key', apiKey);
      apiClient.setApiKey(apiKey);
      setIsAuthenticated(true);
    } catch (error) {
      console.error('Failed to save API key:', error);
      throw error;
    }
  };

  const handleLogout = async () => {
    try {
      await SecureStore.deleteItemAsync('api_key');
      apiClient.clearApiKey();
      setIsAuthenticated(false);
    } catch (error) {
      console.error('Failed to logout:', error);
    }
  };

  if (isLoading) {
    return null; // Or a loading spinner
  }

  if (!isAuthenticated) {
    return <LoginScreen onLogin={handleLogin} />;
  }

  return (
    <NavigationContainer>
      <StatusBar style="light" />
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused, color, size }) => {
            let iconName: keyof typeof Ionicons.glyphMap;

            if (route.name === 'Dashboard') {
              iconName = focused ? 'stats-chart' : 'stats-chart-outline';
            } else if (route.name === 'Portfolio') {
              iconName = focused ? 'wallet' : 'wallet-outline';
            } else if (route.name === 'Signals') {
              iconName = focused ? 'flash' : 'flash-outline';
            } else if (route.name === 'Settings') {
              iconName = focused ? 'settings' : 'settings-outline';
            } else {
              iconName = 'help-circle-outline';
            }

            return <Ionicons name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: '#10B981',
          tabBarInactiveTintColor: '#6B7280',
          tabBarStyle: {
            backgroundColor: '#1F2937',
            borderTopColor: '#374151',
          },
          headerStyle: {
            backgroundColor: '#1F2937',
          },
          headerTintColor: '#fff',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        })}
      >
        <Tab.Screen name="Dashboard" component={DashboardScreen} />
        <Tab.Screen name="Portfolio" component={PortfolioScreen} />
        <Tab.Screen name="Signals" component={SignalsScreen} />
        <Tab.Screen
          name="Settings"
          children={() => <SettingsScreen onLogout={handleLogout} />}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
