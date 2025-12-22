import 'react-native-gesture-handler';
import { Stack } from 'expo-router';
import { AuthProvider } from '../contexts/AuthContext';
import { LanguageProvider } from '../contexts/LanguageContext';
import { UserSettingsProvider } from '../contexts/UserSettingsContext';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import '../i18n';

export default function Layout() {
    return (
        <GestureHandlerRootView style={{ flex: 1 }}>
            <LanguageProvider>
                <AuthProvider>
                    <UserSettingsProvider>
                        <Stack screenOptions={{ headerShown: false }} />
                    </UserSettingsProvider>
                </AuthProvider>
            </LanguageProvider>
        </GestureHandlerRootView>
    );
}