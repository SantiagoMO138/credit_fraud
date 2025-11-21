import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse

class RealisticTransactionGenerator:
    """
    Genera variaciones realistas respetando correlaciones y distribuciones
    """
    
    def __init__(self, df_original):
        self.df = df_original.copy()
        self.feature_cols = [col for col in df_original.columns if col not in ['Class', 'Time']]
        
        # Separar transacciones normales y fraudulentas
        self.df_normal = df_original[df_original['Class'] == 0].copy()
        self.df_fraud = df_original[df_original['Class'] == 1].copy()
        
        print(f"📊 Dataset cargado:")
        print(f"   Normal: {len(self.df_normal):,}")
        print(f"   Fraude: {len(self.df_fraud):,}")
    
    def method_1_correlated_noise(self, percent=0.05):
        """
        Método 1: Ruido correlacionado
        Usa la matriz de covarianza para generar ruido que respeta correlaciones
        """
        print(f"\n🔧 Método 1: Ruido Correlacionado ({percent*100}%)")
        
        df_varied = self.df.copy()
        
        for class_label, df_class in [(0, self.df_normal), (1, self.df_fraud)]:
            if len(df_class) == 0:
                continue
            
            # Obtener índices de esta clase
            class_indices = df_class.index
            
            # Extraer features
            X = df_class[self.feature_cols].values
            
            # Calcular matriz de covarianza
            cov_matrix = np.cov(X.T)
            
            # Generar ruido correlacionado
            noise = np.random.multivariate_normal(
                mean=np.zeros(len(self.feature_cols)),
                cov=cov_matrix * (percent ** 2),
                size=len(X)
            )
            
            # Aplicar ruido
            X_varied = X + noise
            
            # Para Amount, asegurar que sea positivo
            if 'Amount' in self.feature_cols:
                amount_idx = self.feature_cols.index('Amount')
                X_varied[:, amount_idx] = np.maximum(0, X_varied[:, amount_idx])
            
            # Actualizar dataframe
            df_varied.loc[class_indices, self.feature_cols] = X_varied
        
        return df_varied
    
    def method_2_interpolation(self, n_samples=10000):
        """
        Método 2: Interpolación entre transacciones similares
        Crea nuevas transacciones interpolando entre transacciones reales
        """
        print(f"\n🔧 Método 2: Interpolación ({n_samples:,} muestras)")
        
        df_list = []
        
        for class_label, df_class in [(0, self.df_normal), (1, self.df_fraud)]:
            if len(df_class) == 0:
                continue
            
            n_class_samples = int(n_samples * len(df_class) / len(self.df))
            
            X = df_class[self.feature_cols].values
            
            # Para cada muestra, seleccionar dos transacciones aleatorias
            idx1 = np.random.randint(0, len(X), n_class_samples)
            idx2 = np.random.randint(0, len(X), n_class_samples)
            
            # Factor de interpolación aleatorio
            alpha = np.random.beta(2, 2, n_class_samples).reshape(-1, 1)
            
            # Interpolar
            X_interpolated = alpha * X[idx1] + (1 - alpha) * X[idx2]
            
            # Crear dataframe
            df_interpolated = pd.DataFrame(X_interpolated, columns=self.feature_cols)
            df_interpolated['Class'] = class_label
            df_interpolated['Time'] = np.arange(len(df_interpolated))
            
            df_list.append(df_interpolated)
        
        df_varied = pd.concat(df_list, ignore_index=True)
        
        # Mezclar
        df_varied = df_varied.sample(frac=1, random_state=42).reset_index(drop=True)
        df_varied['Time'] = np.arange(len(df_varied))
        
        return df_varied
    
    def method_3_neighbor_sampling(self, n_neighbors=5, noise_factor=0.1):
        """
        Método 3: Sampling basado en vecinos
        Crea variaciones basadas en vecinos cercanos en el espacio de features
        """
        print(f"\n🔧 Método 3: Neighbor Sampling (k={n_neighbors})")
        
        from sklearn.neighbors import NearestNeighbors
        
        df_varied = self.df.copy()
        
        for class_label, df_class in [(0, self.df_normal), (1, self.df_fraud)]:
            if len(df_class) == 0:
                continue
            
            class_indices = df_class.index
            X = df_class[self.feature_cols].values
            
            # Encontrar vecinos cercanos
            nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, len(X)), algorithm='ball_tree')
            nbrs.fit(X)
            
            # Para cada transacción, tomar promedios ponderados con vecinos
            X_varied = []
            for i in range(len(X)):
                distances, indices = nbrs.kneighbors([X[i]])
                
                # Pesos inversamente proporcionales a la distancia
                weights = 1 / (distances[0] + 1e-6)
                weights = weights / weights.sum()
                
                # Promedio ponderado
                x_new = np.average(X[indices[0]], weights=weights, axis=0)
                
                # Agregar ruido pequeño
                noise = np.random.normal(0, noise_factor * X.std(axis=0), X.shape[1])
                x_new = x_new + noise
                
                X_varied.append(x_new)
            
            X_varied = np.array(X_varied)
            
            # Para Amount, asegurar positivo
            if 'Amount' in self.feature_cols:
                amount_idx = self.feature_cols.index('Amount')
                X_varied[:, amount_idx] = np.maximum(0, X_varied[:, amount_idx])
            
            df_varied.loc[class_indices, self.feature_cols] = X_varied
        
        return df_varied
    
    def method_4_temporal_drift(self, drift_rate=0.001):
        """
        Método 4: Drift temporal
        Simula cambios graduales en las distribuciones a lo largo del tiempo
        """
        print(f"\n🔧 Método 4: Temporal Drift (rate={drift_rate})")
        
        df_varied = self.df.copy()
        X = df_varied[self.feature_cols].values
        
        # Aplicar drift lineal
        n_samples = len(X)
        for i, col in enumerate(self.feature_cols):
            # Drift diferente para cada feature
            drift = np.linspace(0, drift_rate * X[:, i].std() * n_samples, n_samples)
            X[:, i] = X[:, i] + drift
        
        df_varied[self.feature_cols] = X
        
        # Asegurar Amount positivo
        if 'Amount' in df_varied.columns:
            df_varied['Amount'] = df_varied['Amount'].clip(lower=0)
        
        return df_varied
    
    def method_5_pca_perturbation(self, n_components=10, perturbation_std=0.5):
        """
        Método 5: Perturbación en espacio PCA
        Varía en el espacio de componentes principales
        """
        print(f"\n🔧 Método 5: PCA Perturbation (components={n_components})")
        
        df_varied = self.df.copy()
        
        for class_label, df_class in [(0, self.df_normal), (1, self.df_fraud)]:
            if len(df_class) == 0:
                continue
            
            class_indices = df_class.index
            X = df_class[self.feature_cols].values
            
            # Estandarizar
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # PCA
            pca = PCA(n_components=min(n_components, X.shape[1]))
            X_pca = pca.fit_transform(X_scaled)
            
            # Perturbar en espacio PCA
            noise = np.random.normal(0, perturbation_std, X_pca.shape)
            X_pca_varied = X_pca + noise
            
            # Reconstruir
            X_scaled_varied = pca.inverse_transform(X_pca_varied)
            X_varied = scaler.inverse_transform(X_scaled_varied)
            
            # Amount positivo
            if 'Amount' in self.feature_cols:
                amount_idx = self.feature_cols.index('Amount')
                X_varied[:, amount_idx] = np.maximum(0, X_varied[:, amount_idx])
            
            df_varied.loc[class_indices, self.feature_cols] = X_varied
        
        return df_varied
    
    def method_6_conditional_noise(self, bins=5):
        """
        Método 6: Ruido condicional según bins de Amount
        Diferentes niveles de ruido según el monto de la transacción
        """
        print(f"\n🔧 Método 6: Conditional Noise (bins={bins})")
        
        df_varied = self.df.copy()
        
        # Crear bins basados en Amount
        df_varied['amount_bin'] = pd.qcut(df_varied['Amount'], q=bins, labels=False, duplicates='drop')
        
        for bin_id in df_varied['amount_bin'].unique():
            bin_mask = df_varied['amount_bin'] == bin_id
            X = df_varied.loc[bin_mask, self.feature_cols].values
            
            # Ruido proporcional al bin (transacciones grandes = más variación)
            noise_factor = (bin_id + 1) / bins * 0.1
            
            for i, col in enumerate(self.feature_cols):
                if col != 'Amount':
                    noise = np.random.normal(0, noise_factor * X[:, i].std(), X.shape[0])
                    df_varied.loc[bin_mask, col] = X[:, i] + noise
        
        df_varied = df_varied.drop('amount_bin', axis=1)
        
        return df_varied


def main():
    parser = argparse.ArgumentParser(description='Generador realista de transacciones')
    parser.add_argument('--input', default='creditcard.csv', help='Archivo de entrada')
    parser.add_argument('--output', default='creditcard_realistic.csv', help='Archivo de salida')
    parser.add_argument('--method', type=int, default=1, choices=[1,2,3,4,5,6],
                       help='Método: 1=Correlated, 2=Interpolation, 3=Neighbors, 4=Drift, 5=PCA, 6=Conditional')
    parser.add_argument('--percent', type=float, default=5.0, help='Porcentaje de variación')
    parser.add_argument('--samples', type=int, default=None, help='Número de muestras (método 2)')
    parser.add_argument('--shuffle', action='store_true', help='Mezclar al final')
    
    args = parser.parse_args()
    
    print("🔬 GENERADOR REALISTA DE TRANSACCIONES")
    print("=" * 60)
    
    # Validar entrada
    if not Path(args.input).exists():
        print(f"❌ Error: No se encontró {args.input}")
        return
    
    # Cargar dataset
    print(f"\n📥 Cargando: {args.input}")
    df = pd.read_csv(args.input)
    print(f"✅ Cargado: {df.shape[0]:,} transacciones")
    
    # Crear generador
    generator = RealisticTransactionGenerator(df)
    
    # Aplicar método seleccionado
    percent_decimal = args.percent / 100.0
    
    if args.method == 1:
        df_result = generator.method_1_correlated_noise(percent_decimal)
    elif args.method == 2:
        n_samples = args.samples if args.samples else len(df)
        df_result = generator.method_2_interpolation(n_samples)
    elif args.method == 3:
        df_result = generator.method_3_neighbor_sampling(noise_factor=percent_decimal)
    elif args.method == 4:
        df_result = generator.method_4_temporal_drift(drift_rate=percent_decimal)
    elif args.method == 5:
        df_result = generator.method_5_pca_perturbation(perturbation_std=percent_decimal * 10)
    elif args.method == 6:
        df_result = generator.method_6_conditional_noise()
    
    # Shuffle si se solicita
    if args.shuffle:
        print(f"\n🔀 Mezclando transacciones...")
        df_result = df_result.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Actualizar Time
    df_result['Time'] = np.arange(len(df_result))
    
    # Estadísticas
    print(f"\n📊 ESTADÍSTICAS:")
    print(f"   Transacciones generadas: {len(df_result):,}")
    
    fraud_original = df['Class'].sum()
    fraud_result = df_result['Class'].sum()
    print(f"\n🔍 Distribución de fraude:")
    print(f"   Original: {fraud_original} ({fraud_original/len(df)*100:.3f}%)")
    print(f"   Resultado: {fraud_result} ({fraud_result/len(df_result)*100:.3f}%)")
    
    # Comparación de estadísticas
    print(f"\n📈 Comparación de features:")
    for col in ['V1', 'V2', 'Amount']:
        orig_mean = df[col].mean()
        orig_std = df[col].std()
        result_mean = df_result[col].mean()
        result_std = df_result[col].std()
        
        diff_mean = abs(result_mean - orig_mean) / abs(orig_mean) * 100
        diff_std = abs(result_std - orig_std) / abs(orig_std) * 100
        
        print(f"   {col:8} | Mean: {diff_mean:5.2f}% diff | Std: {diff_std:5.2f}% diff")
    
    # Guardar
    print(f"\n💾 Guardando: {args.output}")
    df_result.to_csv(args.output, index=False)
    print(f"✅ Guardado exitosamente")
    
    print(f"\n🎯 LISTO!")
    print(f"   Método usado: {args.method}")
    print(f"   Archivo generado: {args.output}")


if __name__ == "__main__":
    main()