# streamlit run portfolio_optimizer.py

import yfinance as yf
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import List, Tuple
import random
import time
from datetime import datetime, timedelta

class PortfolioOptimizer:
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        """
        Inicjalizacja optymalizatora portfela.
        
        :param tickers: Lista symboli giełdowych
        :param start_date: Data początkowa pobierania danych
        :param end_date: Data końcowa pobierania danych
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
        # Pierwsza próba - pobranie danych z Yahoo Finance
        self.data = self._fetch_stock_data()
        
        # Sprawdzenie czy udało się pobrać dane
        if self.data.empty or self.data.shape[1] == 0:
            st.warning("Nie udało się pobrać danych z Yahoo Finance. Używam danych demonstracyjnych.")
            self.data = self._generate_demo_data()
        
        self.returns = self._calculate_returns()
    
    def _fetch_stock_data(self) -> pd.DataFrame:
        """
        Pobieranie danych giełdowych dla wybranych akcji z obsługą błędów.
        
        :return: DataFrame z cenami zamknięcia
        """
        try:
            # Dodajemy opóźnienie między żądaniami, aby uniknąć ograniczeń API
            time.sleep(0.5)
            
            # Pobieramy dane z większym przedziałem czasowym na wypadek problemów
            extended_start = (pd.to_datetime(self.start_date) - pd.DateOffset(days=30)).strftime('%Y-%m-%d')
            
            # Próba pobrania danych
            data = yf.download(
                self.tickers, 
                start=extended_start, 
                end=self.end_date,
                progress=False,
                group_by='ticker'
            )
            
            data.index = pd.to_datetime(data.index)
            
            # Jeśli dane zostały pobrane jako MultiIndex DataFrame
            if isinstance(data.columns, pd.MultiIndex):
                # Wybieramy tylko ceny zamknięcia dla każdego tickera
                close_prices = pd.DataFrame()
                
                for ticker in self.tickers:
                    if ticker in data.columns:
                        try:
                            ticker_data = data[ticker]['Adj Close']
                            close_prices[ticker] = ticker_data
                        except KeyError:
                            st.warning(f"Nie znaleziono danych dla {ticker}")
                    else:
                        st.warning(f"Nie udało się pobrać danych dla {ticker}")
                
                data = close_prices
            else:
                # Jeśli dane są w zwykłym DataFrame, wybieramy tylko ceny zamknięcia
                data = data['Adj Close']
            
            # Przycinamy dane do żądanego przedziału czasowego
            data = data.loc[self.start_date:self.end_date]
            
            return data
        
        except yf.shared._exceptions.TzMissingError as e:
            st.error(f"Błąd podczas pobierania danych: {str(e)}. Możliwe, że niektóre tickery zostały wycofane.")
            return pd.DataFrame()
        except ValueError as e:
            st.error(f"Błąd podczas pobierania danych: {str(e)}. Sprawdź poprawność symboli giełdowych.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Błąd podczas pobierania danych: {str(e)}")
            return pd.DataFrame()
    
    def _generate_demo_data(self) -> pd.DataFrame:
        """
        Generowanie danych demonstracyjnych w przypadku problemów z API.
        
        :return: DataFrame z symulowanymi cenami
        """
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        
        days = (end_date - start_date).days
        date_range = pd.date_range(start=start_date, periods=days)
        
        data = pd.DataFrame(index=date_range)
        
        # Dla każdego tickera generujemy realistyczne ceny
        for ticker in self.tickers:
            # Losowy początkowy kurs
            start_price = random.uniform(50.0, 500.0)
            
            # Symulacja ruchu cen akcji (random walk)
            price_series = [start_price]
            for _ in range(len(date_range) - 1):
                # Dzienne zmiany ceny z odchyleniem 1-2%
                daily_return = random.normalvariate(0.0005, 0.015)
                next_price = price_series[-1] * (1 + daily_return)
                price_series.append(next_price)
            
            data[ticker] = price_series
        
        return data
    
    def _calculate_returns(self) -> pd.DataFrame:
        """
        Obliczanie dziennych stóp zwrotu.
        
        :return: DataFrame ze stopami zwrotu
        """
        if self.data.empty:
            # Jeśli brak danych, zwróć pusty DataFrame
            return pd.DataFrame()
        
        returns = self.data.pct_change().dropna()
        
        # Sprawdzenie, czy zwrócone dane nie zawierają wartości NaN
        if returns.isnull().values.any():
            # Zastąp wartości NaN średnią dla danej kolumny
            returns = returns.fillna(returns.mean())
        
        return returns
    
    def calculate_portfolio_metrics(self) -> dict:
        """
        Obliczanie kluczowych metryk portfela z obsługą błędów.
        
        :return: Słownik z metrykami portfela
        """
        if self.returns.empty:
            # Jeśli brak danych, zwróć puste lub zerowe metryki
            n = len(self.tickers)
            return {
                'mean_returns': np.zeros(n),
                'covariance_matrix': np.zeros((n, n)),
                'annual_returns': np.zeros(n),
                'annual_volatility': np.zeros(n)
            }
        
        try:
            mean_returns = self.returns.mean()
            cov_matrix = self.returns.cov()
            
            return {
                'mean_returns': mean_returns,
                'covariance_matrix': cov_matrix,
                'annual_returns': mean_returns * 252,
                'annual_volatility': np.sqrt(np.diag(cov_matrix) * 252)
            }
        except Exception as e:
            st.error(f"Błąd podczas obliczania metryk portfela: {str(e)}")
            n = len(self.tickers)
            return {
                'mean_returns': np.zeros(n),
                'covariance_matrix': np.zeros((n, n)),
                'annual_returns': np.zeros(n),
                'annual_volatility': np.zeros(n)
            }
    
    def optimize_portfolio(self, risk_free_rate: float = 0.02) -> Tuple[np.ndarray, float, float]:
        """
        Optymalizacja portfela przy użyciu modelu Markowitza z obsługą błędów.
        
        :param risk_free_rate: Stopa wolna od ryzyka
        :return: Optymalne wagi, oczekiwany zwrot, ryzyko
        """
        metrics = self.calculate_portfolio_metrics()
        mean_returns = metrics['mean_returns']
        cov_matrix = metrics['covariance_matrix']
        
        n = len(self.tickers)
        
        # Sprawdzenie poprawności danych wejściowych
        if np.isnan(mean_returns).any() or np.isnan(cov_matrix.values).any():
            st.warning("Wykryto braki danych. Używam danych zastępczych dla optymalizacji.")
            mean_returns = np.ones(n) * 0.001  # 0.1% jako domyślny zwrot
            cov_matrix = np.eye(n) * 0.01  # 1% jako domyślna zmienność
        
        try:
            weights = cp.Variable(n)
            ret = cp.sum(cp.multiply(mean_returns, weights))
            risk = cp.quad_form(weights, cov_matrix)
            
            constraints = [
                cp.sum(weights) == 1,
                weights >= 0
            ]
            
            # Definiujemy problem optymalizacji - maksymalizacja wskaźnika Sharpe'a
            objective = cp.Maximize(ret - risk_free_rate * risk)
            prob = cp.Problem(objective, constraints)
            
            try:
                prob.solve()
                
                if prob.status != 'optimal':
                    raise Exception("Nie udało się znaleźć optymalnego rozwiązania")
                
                optimal_weights = weights.value
                expected_return = mean_returns @ optimal_weights
                portfolio_risk = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
                
                return optimal_weights, expected_return, portfolio_risk
            
            except Exception as e:
                st.warning(f"Problem z optymalizacją: {str(e)}. Używam równych wag.")
                # Jeśli optymalizacja nie powiodła się, użyj równych wag
                equal_weights = np.ones(n) / n
                expected_return = mean_returns @ equal_weights
                portfolio_risk = np.sqrt(equal_weights @ cov_matrix @ equal_weights)
                
                return equal_weights, expected_return, portfolio_risk
                
        except Exception as e:
            st.error(f"Błąd podczas optymalizacji portfela: {str(e)}")
            # Zwróć równy podział w przypadku błędu
            equal_weights = np.ones(n) / n
            return equal_weights, 0.0, 0.0
    
    def calculate_efficient_frontier(self, points: int = 20) -> pd.DataFrame:
        """
        Obliczanie punktów na efektywnej granicy portfela.
        
        :param points: Liczba punktów na krzywej
        :return: DataFrame z punktami efektywnej granicy
        """
        metrics = self.calculate_portfolio_metrics()
        mean_returns = metrics['mean_returns']
        cov_matrix = metrics['covariance_matrix']
        
        n = len(self.tickers)
        
        # Przygotowanie wyników
        results = []
        
        # Minimalizacja ryzyka
        weights = cp.Variable(n)
        risk = cp.quad_form(weights, cov_matrix)
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]
        
        # Znajdź portfel o minimalnym ryzyku
        objective = cp.Minimize(risk)
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve()
            min_risk_weights = weights.value
            min_risk = np.sqrt(risk.value)
            min_return = mean_returns @ min_risk_weights
            
            # Znajdź portfel o maksymalnym zwrocie
            ret = cp.sum(cp.multiply(mean_returns, weights))
            objective = cp.Maximize(ret)
            prob = cp.Problem(objective, constraints)
            prob.solve()
            max_return_weights = weights.value
            max_return = mean_returns @ max_return_weights
            max_risk = np.sqrt(cp.quad_form(weights, cov_matrix).value)
            
            # Oblicz punkty efektywnej granicy
            target_returns = np.linspace(min_return, max_return, points)
            for target_return in target_returns:
                constraints = [
                    cp.sum(weights) == 1,
                    weights >= 0,
                    cp.sum(cp.multiply(mean_returns, weights)) == target_return
                ]
                
                objective = cp.Minimize(risk)
                prob = cp.Problem(objective, constraints)
                prob.solve()
                
                if prob.status == 'optimal':
                    optimal_weights = weights.value
                    portfolio_risk = np.sqrt(risk.value)
                    sharpe_ratio = (target_return - 0.02) / portfolio_risk if portfolio_risk > 0 else 0
                    
                    results.append({
                        'return': target_return,
                        'risk': portfolio_risk,
                        'sharpe_ratio': sharpe_ratio,
                        'weights': optimal_weights
                    })
            
            return pd.DataFrame(results)
        
        except Exception as e:
            st.warning(f"Nie udało się obliczyć efektywnej granicy: {str(e)}")
            return pd.DataFrame(columns=['return', 'risk', 'sharpe_ratio', 'weights'])
    
    def monte_carlo_simulation(self, num_simulations: int = 5000) -> pd.DataFrame:
        """
        Symulacja Monte Carlo dla portfela.
        
        :param num_simulations: Liczba symulacji
        :return: DataFrame z wynikami symulacji
        """
        try:
            metrics = self.calculate_portfolio_metrics()
            mean_returns = metrics['annual_returns']
            cov_matrix = metrics['covariance_matrix'] * 252  # Konwersja na roczną kowariancję
            
            n = len(self.tickers)
            
            # Przygotowanie na wyniki symulacji
            results = []
            
            for _ in range(num_simulations):
                # Losowe wagi sumujące się do 1
                weights = np.random.random(n)
                weights = weights / np.sum(weights)
                
                # Obliczenie oczekiwanego zwrotu i ryzyka
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                sharpe_ratio = (portfolio_return - 0.02) / portfolio_risk if portfolio_risk > 0 else 0
                
                results.append({
                    'return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio,
                    'weights': weights
                })
            
            return pd.DataFrame(results)
        
        except Exception as e:
            st.error(f"Błąd podczas symulacji Monte Carlo: {str(e)}")
            return pd.DataFrame(columns=['return', 'risk', 'sharpe_ratio', 'weights'])

    def calculate_var(self, weights: np.ndarray, confidence_level: float = 0.95, days: int = 1) -> float:
        """
        Obliczanie wartości zagrożonej (Value at Risk) dla portfela.
        
        :param weights: Wagi aktywów w portfelu
        :param confidence_level: Poziom ufności dla VaR
        :param days: Horyzont czasowy w dniach
        :return: Wartość VaR
        """
        try:
            # Obliczanie zwrotów portfela na podstawie danych historycznych
            portfolio_returns = self.returns.dot(weights)
            
            # Sortowanie zwrotów
            sorted_returns = np.sort(portfolio_returns)
            
            # Indeks dla danego poziomu ufności
            index = int((1 - confidence_level) * len(sorted_returns))
            
            # VaR to ujemna wartość kwantyla
            var = -sorted_returns[index]
            
            # Skalowanie VaR do określonego horyzontu czasowego
            var_scaled = var * np.sqrt(days)
            
            return var_scaled
        
        except Exception as e:
            st.error(f"Błąd podczas obliczania VaR: {str(e)}")
            return 0.0
    
    def calculate_cvar(self, weights: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Obliczanie warunkowej wartości zagrożonej (Conditional Value at Risk) dla portfela.
        
        :param weights: Wagi aktywów w portfelu
        :param confidence_level: Poziom ufności dla CVaR
        :return: Wartość CVaR
        """
        try:
            # Obliczanie zwrotów portfela na podstawie danych historycznych
            portfolio_returns = self.returns.dot(weights)
            
            # Sortowanie zwrotów
            sorted_returns = np.sort(portfolio_returns)
            
            # Indeks dla danego poziomu ufności
            index = int((1 - confidence_level) * len(sorted_returns))
            
            # CVaR to średnia strat poniżej VaR
            cvar = -np.mean(sorted_returns[:index])
            
            return cvar
        
        except Exception as e:
            st.error(f"Błąd podczas obliczania CVaR: {str(e)}")
            return 0.0

def main():
    st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")
    
    st.title('Portfolio Optimizer 📈')
    st.markdown("""
    Narzędzie do analizy i optymalizacji portfela inwestycyjnego na podstawie historycznych danych giełdowych.
    """)
    
    # Konfiguracja bocznego menu
    with st.sidebar:
        st.header('Konfiguracja Portfela')
        
        # Wybór aktywów
        tickers_input = st.text_input(
            'Symbole aktywów (oddzielone przecinkami)', 
            'AAPL,MSFT,GOOGL,AMZN,TSLA,WIG20.WA,WIG30.WA'
        )
        
        # Daty
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                'Data początkowa', 
                value=datetime.now() - timedelta(days=365*3)
            )
        with col2:
            end_date = st.date_input(
                'Data końcowa', 
                value=datetime.now()
            )
        
        # Parametry optymalizacji
        risk_free_rate = st.slider(
            'Stopa wolna od ryzyka (%)', 
            min_value=0.0, 
            max_value=5.0, 
            value=2.0, 
            step=0.1
        ) / 100
        
        # Parametry symulacji
        num_simulations = st.slider(
            'Liczba symulacji Monte Carlo', 
            min_value=1000, 
            max_value=10000, 
            value=5000, 
            step=1000
        )
        
        # Parametry ryzyka
        confidence_level = st.slider(
            'Poziom ufności dla VaR/CVaR (%)', 
            min_value=90, 
            max_value=99, 
            value=95, 
            step=1
        ) / 100
    
    # Przetwarzanie listy aktywów
    tickers_list = [ticker.strip() for ticker in tickers_input.split(',')]
    
    # Tworzenie optymalizatora
    with st.spinner('Pobieranie danych i inicjalizacja optymalizatora...'):
        optimizer = PortfolioOptimizer(tickers_list, start_date, end_date)
    
    # Sprawdzenie, czy mamy dane
    if optimizer.data.empty:
        st.error("Nie udało się pobrać danych. Sprawdź poprawność symboli giełdowych.")
        return
    
    # Zakładki z różnymi analizami
    tab1, tab2, tab3, tab4 = st.tabs([
        "Optymalizacja Portfela", 
        "Analiza Ryzyka", 
        "Symulacje Monte Carlo",
        "Dane Historyczne"
    ])
    
    with tab1:
        st.header("Optymalizacja Portfela")
        
        # Optymalizacja portfela
        with st.spinner('Optymalizacja portfela...'):
            weights, expected_return, portfolio_risk = optimizer.optimize_portfolio(risk_free_rate)
            
            # Wskaźnik Sharpe'a
            sharpe_ratio = (expected_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            # Wyświetlenie wyników
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Oczekiwany roczny zwrot", f"{expected_return*100:.2f}%")
            with col2:
                st.metric("Ryzyko (odchylenie standardowe)", f"{portfolio_risk*100:.2f}%")
            with col3:
                st.metric("Wskaźnik Sharpe'a", f"{sharpe_ratio:.2f}")
        
        # Wizualizacja optymalnych wag
        st.subheader("Optymalne wagi aktywów")
        
        fig = px.pie(
            names=tickers_list,
            values=weights,
            title="Podział portfela"
        )
        st.plotly_chart(fig)
        
        # Tabela z wagami
        weights_df = pd.DataFrame({
            'Aktywo': tickers_list,
            'Waga (%)': [f"{w*100:.2f}%" for w in weights]
        })
        st.table(weights_df)
        
        # Efektywna granica
        st.subheader("Efektywna granica portfela")
        
        with st.spinner('Obliczanie efektywnej granicy...'):
            ef_data = optimizer.calculate_efficient_frontier()
            mc_data = optimizer.monte_carlo_simulation(num_simulations // 5)  # Mniejsza liczba dla lepszej wydajności
            
            if not ef_data.empty and not mc_data.empty:
                # Wykres efektywnej granicy
                fig = go.Figure()
                
                # Dodanie punktów z symulacji Monte Carlo
                fig.add_trace(go.Scatter(
                    x=mc_data['risk'],
                    y=mc_data['return'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=mc_data['sharpe_ratio'],
                        colorscale='Viridis',
                        colorbar=dict(title="Wskaźnik Sharpe'a"),
                        showscale=True
                    ),
                    name="Symulacje losowe"
                ))
                
                # Dodanie linii efektywnej granicy
                fig.add_trace(go.Scatter(
                    x=ef_data['risk'],
                    y=ef_data['return'],
                    mode='lines+markers',
                    line=dict(color='red', width=4),
                    name="Efektywna granica"
                ))
                
                # Dodanie punktu optymalnego portfela
                fig.add_trace(go.Scatter(
                    x=[portfolio_risk],
                    y=[expected_return],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='gold',
                        line=dict(width=2, color='black')
                    ),
                    name="Optymalny portfel"
                ))
                
                # Ustawienia wykresu
                fig.update_layout(
                    title="Efektywna granica portfela",
                    xaxis_title="Ryzyko (odchylenie standardowe)",
                    yaxis_title="Oczekiwany zwrot",
                    hovermode="closest",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Nie udało się wygenerować wykresu efektywnej granicy.")
    
    with tab2:
        st.header("Analiza Ryzyka")
        
        # Obliczenie miar ryzyka
        with st.spinner('Obliczanie miar ryzyka...'):
            var_value = optimizer.calculate_var(weights, confidence_level)
            cvar_value = optimizer.calculate_cvar(weights, confidence_level)
            
            # Wyświetlenie wyników
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    f"Value at Risk (VaR) - poziom ufności {confidence_level*100:.0f}%", 
                    f"{var_value*100:.2f}%"
                )
                st.markdown(f"""
                **Interpretacja:** Z prawdopodobieństwem {confidence_level*100:.0f}%, dzienna strata 
                portfela nie przekroczy {var_value*100:.2f}% jego wartości.
                """)
            with col2:
                st.metric(
                    f"Conditional VaR (CVaR/Expected Shortfall)", 
                    f"{cvar_value*100:.2f}%"
                )
                st.markdown(f"""
                **Interpretacja:** Jeśli strata przekroczy VaR, to oczekiwana wartość tej 
                straty wyniesie {cvar_value*100:.2f}% wartości portfela.
                """)
            
            # Korelacje między aktywami
            st.subheader("Korelacje między aktywami")
            
            corr_matrix = optimizer.returns.corr()
            
            fig = px.imshow(
                corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Mapa korelacji aktywów"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Wykres zmienności aktywów
            st.subheader("Zmienność aktywów")
            
            metrics = optimizer.calculate_portfolio_metrics()
            volatility = pd.Series(metrics['annual_volatility'] * 100, index=tickers_list)
            returns = pd.Series(metrics['annual_returns'] * 100, index=tickers_list)
            
            risk_return_df = pd.DataFrame({
                'Aktywo': tickers_list,
                'Zmienność (%)': volatility,
                'Średni zwrot (%)': returns
            }).sort_values('Zmienność (%)', ascending=False)
            
            fig = px.bar(
                risk_return_df,
                x='Aktywo',
                y='Zmienność (%)',
                color='Średni zwrot (%)',
                color_continuous_scale="RdYlGn",
                title="Zmienność i zwrot poszczególnych aktywów"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Symulacje Monte Carlo")
        
        # Symulacje przyszłych wartości portfela
        st.subheader("Symulacja przyszłych wartości portfela")
        
        # Parametry symulacji
        col1, col2 = st.columns(2)
        with col1:
            initial_investment = st.number_input(
                'Początkowa wartość inwestycji', 
                min_value=1000, 
                value=100000,
                step=10000
            )
        with col2:
            years = st.slider(
                'Okres symulacji (lata)', 
                min_value=1, 
                max_value=30, 
                value=10
            )
        
        with st.spinner('Wykonywanie symulacji Monte Carlo...'):
            # Parametry rynkowe
            metrics = optimizer.calculate_portfolio_metrics()
            annual_return = (weights @ metrics['annual_returns'])
            annual_volatility = np.sqrt(weights @ metrics['covariance_matrix'] @ weights) * np.sqrt(252)
            
            # Parametry symulacji
            days = 252 * years
            daily_return = annual_return / 252
            daily_volatility = annual_volatility / np.sqrt(252)
            
            # Przygotowanie na wyniki symulacji
            simulation_results = []
            
            for _ in range(num_simulations // 10):  # Zmniejszona liczba dla lepszej wydajności
                # Symulacja ścieżki cen
                price_path = [initial_investment]
                
                for _ in range(days):
                    # Random walk z dryfem
                    daily_return_sample = np.random.normal(daily_return, daily_volatility)
                    price_path.append(price_path[-1] * (1 + daily_return_sample))
                
                simulation_results.append(price_path)
            
            # Konwersja wyników do DataFrame
            sim_df = pd.DataFrame(simulation_results).T
            
            # Obliczenie kwantyli
            quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
            quantile_data = np.percentile(sim_df, q=[q * 100 for q in quantiles], axis=1)
            
            # Tworzenie DataFrame dla wykresów
            sim_dates = pd.date_range(start=datetime.now(), periods=len(sim_df), freq='B')
            quantile_df = pd.DataFrame(data=quantile_data.T, index=sim_dates)
            quantile_df.columns = [f"{q*100:.0f}%" for q in quantiles]
            
            # Wykres symulacji
            fig = go.Figure()
            
            # Dodanie obszarów kwantyli
            fig.add_trace(go.Scatter(
                x=quantile_df.index,
                y=quantile_df['95%'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=quantile_df.index,
                y=quantile_df['5%'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 100, 80, 0.2)',
                name='90% przedział'
            ))
            
            fig.add_trace(go.Scatter(
                x=quantile_df.index,
                y=quantile_df['75%'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=quantile_df.index,
                y=quantile_df['25%'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 100, 80, 0.4)',
                name='50% przedział'
            ))
            
            # Dodanie linii mediany
            fig.add_trace(go.Scatter(
                x=quantile_df.index,
                y=quantile_df['50%'],
                mode='lines',
                line=dict(color='rgb(0, 100, 80)', width=2),
                name='Mediana'
            ))
            
            # Dodanie początkowej inwestycji
            fig.add_trace(go.Scatter(
                x=[quantile_df.index[0]],
                y=[initial_investment],
                mode='markers',
                marker=dict(color='gold', size=10, line=dict(color='black', width=2)),
                name='Początkowa inwestycja'
            ))
            
            # Ustawienia wykresu
            fig.update_layout(
                title=f"Symulacja Monte Carlo wartości portfela na {years} lat",
                xaxis_title="Data",
                yaxis_title="Wartość portfela",
                hovermode="x unified",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statystyki końcowe
            final_values = sim_df.iloc[-1]
            median_final = np.median(final_values)
            mean_final = np.mean(final_values)
            
            # Wyświetlenie wyników
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Mediana końcowej wartości", 
                    f"{median_final:,.2f} PLN",
                    f"{(median_final/initial_investment - 1)*100:.2f}%"
                )
            with col2:
                st.metric(
                    "Średnia końcowa wartość", 
                    f"{mean_final:,.2f} PLN",
                    f"{(mean_final/initial_investment - 1)*100:.2f}%"
                )
            with col3:
                prob_profit = np.mean(final_values > initial_investment) * 100
                st.metric(
                    "Prawdopodobieństwo zysku", 
                    f"{prob_profit:.2f}%"
                )
            
            # Histogram końcowych wartości
            fig = px.histogram(
                final_values,
                nbins=50,
                title="Rozkład końcowych wartości portfela",
                labels={"value": "Końcowa wartość", "count": "Liczba symulacji"},
                marginal="box"
            )
            
            # Dodanie linii dla początkowej inwestycji
            fig.add_vline(
                x=initial_investment,
                line_dash="dash", 
                line_color="red",
                annotation_text="Początkowa inwestycja",
                annotation_position="top"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Dane Historyczne")
        
        # Wykres cen historycznych
        st.subheader("Historyczne ceny aktywów")
        
        # Normalizacja danych do porównania
        normalized_data = optimizer.data / optimizer.data.iloc[0] * 100
        
        fig = px.line(
            normalized_data,
            x=normalized_data.index,
            y=normalized_data.columns,
            title="Znormalizowane ceny aktywów (100 = początek okresu)",
            labels={"value": "Wartość znormalizowana", "variable": "Aktywo"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statystyki opisowe
        st.subheader("Statystyki opisowe")
        
        # Dzienne stopy zwrotu
        daily_returns = optimizer.returns * 100  # Konwersja na procenty
        
        # Tabela ze statystykami
        stats_df = daily_returns.describe().T.sort_values('mean', ascending=False)
        stats_df.columns = ['Liczba obserwacji', 'Średnia (%)', 'Odch. std. (%)', 
                           'Min (%)', '25%', '50%', '75%', 'Max (%)']
        
        st.dataframe(stats_df.style.format({
            'Średnia (%)': '{:.2f}',
            'Odch. std. (%)': '{:.2f}',
            'Min (%)': '{:.2f}',
            '25%': '{:.2f}',
            '50%': '{:.2f}',
            '75%': '{:.2f}',
            'Max (%)': '{:.2f}'
        }))
        
        # Wykres rozkładu dziennych zwrotów
        st.subheader("Rozkład dziennych stóp zwrotu")
        
        # Wybór aktywów do wykresu
        selected_assets = st.multiselect(
            'Wybierz aktywa do analizy',
            options=daily_returns.columns.tolist(),
            default=daily_returns.columns[:3].tolist()
        )
        
        if selected_assets:
            fig = go.Figure()
            
            for asset in selected_assets:
                fig.add_trace(go.Histogram(
                    x=daily_returns[asset],
                    name=asset,
                    opacity=0.7,
                    histnorm='probability',
                    nbinsx=50
                ))
            
            fig.update_layout(
                title="Rozkład dziennych stóp zwrotu",
                xaxis_title="Dzienna stopa zwrotu (%)",
                yaxis_title="Prawdopodobieństwo",
                barmode='overlay',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Surowe dane
        st.subheader("Surowe dane")
        with st.expander("Pokaż dane"):
            st.dataframe(optimizer.data)

if __name__ == "__main__":
    main()
    