import pandas as pd
import numpy as np
from pathlib import Path
import argparse

class BalancedFraudGenerator:
    """
    Genera dataset balanceado con 40% de fraude usando variaciones correlacionadas
    """
    
    def __init__(self, df_original):
        self.df = df_original.copy()
        self.feature_cols = [col for col in df_original.columns if col not in ['Class', 'Time']]
        
        # Separar transacciones
        self.df_normal = df_original[df_original['Class'] == 0].copy()
        self.df_fraud = df_original[df_original['Class'] == 1].copy()
        
        print("📊 GENERADOR DE DATASET BALANCEADO")
        print("=" * 60)
        print(f"\n📈 Dataset original:")
        print(f"   Normal: {len(self.df_normal):,} ({len(self.df_normal)/len(df_original)*100:.2f}%)")
        print(f"   Fraude: {len(self.df_fraud):,} ({len(self.df_fraud)/len(df_original)*100:.2f}%)")
    
    def generate_fraud_variations(self, n_variations, noise_percent=0.05):
        """
        Genera variaciones de transacciones fraudulentas usando ruido correlacionado
        
        Args:
            n_variations: Número de variaciones a generar por cada fraude original
            noise_percent: Porcentaje de ruido a aplicar
        """
        print(f"\n🔧 Generando variaciones de fraude...")
        print(f"   Variaciones por fraude: {n_variations}")
        print(f"   Nivel de ruido: {noise_percent*100}%")
        
        if len(self.df_fraud) == 0:
            raise ValueError("No hay transacciones fraudulentas en el dataset")
        
        # Extraer features de fraudes
        X_fraud = self.df_fraud[self.feature_cols].values
        
        # Calcular matriz de covarianza de los fraudes
        cov_matrix = np.cov(X_fraud.T)
        
        print(f"   📊 Matriz de covarianza calculada: {cov_matrix.shape}")
        
        # Generar variaciones
        fraud_variations = []
        
        for i in range(len(self.df_fraud)):
            # Transacción original
            original_fraud = X_fraud[i]
            
            # Generar n_variations variaciones con ruido correlacionado
            for v in range(n_variations):
                # Ruido correlacionado
                noise = np.random.multivariate_normal(
                    mean=np.zeros(len(self.feature_cols)),
                    cov=cov_matrix * (noise_percent ** 2)
                )
                
                # Aplicar ruido
                fraud_varied = original_fraud + noise
                
                # Asegurar Amount positivo
                if 'Amount' in self.feature_cols:
                    amount_idx = self.feature_cols.index('Amount')
                    fraud_varied[amount_idx] = max(0, fraud_varied[amount_idx])
                
                fraud_variations.append(fraud_varied)
        
        # Convertir a DataFrame
        df_fraud_varied = pd.DataFrame(fraud_variations, columns=self.feature_cols)
        df_fraud_varied['Class'] = 1
        df_fraud_varied['Time'] = np.arange(len(df_fraud_varied))
        
        print(f"   ✅ Generados: {len(df_fraud_varied):,} fraudes sintéticos")
        
        return df_fraud_varied
    
    def create_balanced_dataset(self, target_fraud_rate=0.40, 
                                total_samples=None, 
                                noise_percent=0.05,
                                include_originals=True):
        """
        Crea dataset balanceado con tasa de fraude objetivo
        
        Args:
            target_fraud_rate: Porcentaje objetivo de fraude (0.40 = 40%)
            total_samples: Número total de muestras (None = calcular automáticamente)
            noise_percent: Nivel de ruido para variaciones
            include_originals: Incluir fraudes originales
        """
        
        print(f"\n🎯 OBJETIVO: {target_fraud_rate*100}% de fraude")
        
        # Calcular tamaños
        n_fraud_original = len(self.df_fraud)
        
        if total_samples is None:
            # Calcular basado en mantener todas las transacciones normales
            total_samples = int(len(self.df_normal) / (1 - target_fraud_rate))
        
        n_fraud_needed = int(total_samples * target_fraud_rate)
        n_normal_needed = total_samples - n_fraud_needed
        
        print(f"\n📐 Cálculos:")
        print(f"   Total muestras objetivo: {total_samples:,}")
        print(f"   Fraudes necesarios: {n_fraud_needed:,}")
        print(f"   Normales necesarias: {n_normal_needed:,}")
        
        # Validar que tenemos suficientes normales
        if n_normal_needed > len(self.df_normal):
            print(f"\n⚠️  No hay suficientes transacciones normales")
            print(f"   Ajustando total_samples...")
            total_samples = int(len(self.df_normal) / (1 - target_fraud_rate))
            n_fraud_needed = int(total_samples * target_fraud_rate)
            n_normal_needed = total_samples - n_fraud_needed
            print(f"   Nuevo total: {total_samples:,}")
        
        # Calcular cuántas variaciones generar
        if include_originals:
            n_fraud_synthetic = n_fraud_needed - n_fraud_original
            
            if n_fraud_synthetic < 0:
                # Tenemos más fraudes originales de los necesarios
                print(f"\n✂️  Submuestreando fraudes originales a {n_fraud_needed:,}")
                df_fraud_final = self.df_fraud.sample(n=n_fraud_needed, random_state=42)
                df_fraud_synthetic = pd.DataFrame()
            else:
                # Necesitamos generar fraudes sintéticos
                n_variations = int(np.ceil(n_fraud_synthetic / n_fraud_original))
                
                print(f"\n🔬 Generando fraudes sintéticos:")
                print(f"   Fraudes sintéticos necesarios: {n_fraud_synthetic:,}")
                print(f"   Variaciones por fraude original: {n_variations}")
                
                # Generar variaciones
                df_fraud_synthetic = self.generate_fraud_variations(n_variations, noise_percent)
                
                # Submuestrear si generamos de más
                if len(df_fraud_synthetic) > n_fraud_synthetic:
                    df_fraud_synthetic = df_fraud_synthetic.sample(n=n_fraud_synthetic, random_state=42)
                
                # Combinar con originales
                df_fraud_final = pd.concat([
                    self.df_fraud,
                    df_fraud_synthetic
                ], ignore_index=True)
        else:
            # Solo fraudes sintéticos (sin originales)
            n_variations = int(np.ceil(n_fraud_needed / n_fraud_original))
            print(f"\n🔬 Generando solo fraudes sintéticos:")
            print(f"   Variaciones por fraude original: {n_variations}")
            
            df_fraud_synthetic = self.generate_fraud_variations(n_variations, noise_percent)
            
            if len(df_fraud_synthetic) > n_fraud_needed:
                df_fraud_synthetic = df_fraud_synthetic.sample(n=n_fraud_needed, random_state=42)
            
            df_fraud_final = df_fraud_synthetic
        
        # Submuestrear normales
        print(f"\n✂️  Submuestreando transacciones normales a {n_normal_needed:,}")
        df_normal_sampled = self.df_normal.sample(n=n_normal_needed, random_state=42)
        
        # Combinar y mezclar
        print(f"\n🔀 Combinando y mezclando dataset...")
        df_balanced = pd.concat([
            df_normal_sampled,
            df_fraud_final
        ], ignore_index=True)
        
        # Mezclar
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Actualizar Time
        df_balanced['Time'] = np.arange(len(df_balanced))
        
        # Estadísticas finales
        n_fraud_final = (df_balanced['Class'] == 1).sum()
        n_normal_final = (df_balanced['Class'] == 0).sum()
        fraud_rate_final = n_fraud_final / len(df_balanced)
        
        print(f"\n📊 DATASET FINAL:")
        print(f"   Total: {len(df_balanced):,}")
        print(f"   Normal: {n_normal_final:,} ({n_normal_final/len(df_balanced)*100:.2f}%)")
        print(f"   Fraude: {n_fraud_final:,} ({fraud_rate_final*100:.2f}%)")
        print(f"   Tasa de fraude objetivo: {target_fraud_rate*100:.1f}%")
        print(f"   Tasa de fraude real: {fraud_rate_final*100:.2f}%")
        
        if abs(fraud_rate_final - target_fraud_rate) < 0.01:
            print(f"   ✅ OBJETIVO ALCANZADO")
        else:
            print(f"   ⚠️  Desviación: {abs(fraud_rate_final - target_fraud_rate)*100:.2f}%")
        
        return df_balanced
    
    def analyze_quality(self, df_generated):
        """
        Analiza la calidad de las transacciones generadas
        """
        print(f"\n🔍 ANÁLISIS DE CALIDAD")
        print("=" * 60)
        
        # Separar fraudes originales y sintéticos (si es posible)
        fraud_generated = df_generated[df_generated['Class'] == 1]
        
        # Comparar estadísticas
        print(f"\n📈 Comparación de estadísticas (Fraudes):")
        for col in ['V1', 'V2', 'V10', 'Amount']:
            if col in self.feature_cols:
                orig_mean = self.df_fraud[col].mean()
                orig_std = self.df_fraud[col].std()
                gen_mean = fraud_generated[col].mean()
                gen_std = fraud_generated[col].std()
                
                diff_mean = abs(gen_mean - orig_mean) / abs(orig_mean) * 100
                diff_std = abs(gen_std - orig_std) / abs(orig_std) * 100
                
                print(f"   {col:8} | Mean: {diff_mean:5.2f}% diff | Std: {diff_std:5.2f}% diff")
        
        # Correlaciones
        orig_fraud_corr = self.df_fraud[self.feature_cols].corr().values
        gen_fraud_corr = fraud_generated[self.feature_cols].corr().values
        
        corr_preservation = np.corrcoef(
            orig_fraud_corr.flatten(),
            gen_fraud_corr.flatten()
        )[0, 1]
        
        print(f"\n🔗 Preservación de correlaciones (Fraudes):")
        print(f"   Similitud: {corr_preservation:.4f}")
        
        if corr_preservation > 0.95:
            print(f"   ✅ EXCELENTE - Correlaciones bien preservadas")
        elif corr_preservation > 0.90:
            print(f"   ✅ BUENO - Correlaciones aceptables")
        elif corr_preservation > 0.85:
            print(f"   ⚠️  ACEPTABLE - Algunas correlaciones perdidas")
        else:
            print(f"   ❌ BAJO - Revisar parámetros de generación")


def main():
    parser = argparse.ArgumentParser(
        description='Genera dataset balanceado con 40% de fraude usando variaciones correlacionadas'
    )
    parser.add_argument('--input', default='creditcard.csv', help='Archivo CSV de entrada')
    parser.add_argument('--output', default='creditcard_balanced_40pct.csv', help='Archivo CSV de salida')
    parser.add_argument('--fraud-rate', type=float, default=0.10, 
                       help='Tasa objetivo de fraude (default: 0.10 = 10%%)')
    parser.add_argument('--samples', type=int, default=None,
                       help='Total de muestras (None = calcular automáticamente)')
    parser.add_argument('--noise', type=float, default=0.05,
                       help='Porcentaje de ruido (default: 0.05%%)')
    parser.add_argument('--no-originals', action='store_true',
                       help='No incluir fraudes originales (solo sintéticos)')
    parser.add_argument('--analyze', action='store_true',
                       help='Analizar calidad del dataset generado')
    
    args = parser.parse_args()
    
    print("🔬 GENERADOR DE DATASET BALANCEADO CON VARIACIONES")
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
    generator = BalancedFraudGenerator(df)
    
    # Generar dataset balanceado
    noise_decimal = args.noise / 100.0
    
    df_balanced = generator.create_balanced_dataset(
        target_fraud_rate=args.fraud_rate,
        total_samples=args.samples,
        noise_percent=noise_decimal,
        include_originals=not args.no_originals
    )
    
    # Análisis de calidad
    if args.analyze:
        generator.analyze_quality(df_balanced)
    
    # Guardar
    print(f"\n💾 Guardando: {args.output}")
    df_balanced.to_csv(args.output, index=False)
    print(f"✅ Guardado exitosamente")
    
    print(f"\n🎯 RESUMEN:")
    print(f"   Archivo: {args.output}")
    print(f"   Total: {len(df_balanced):,} transacciones")
    print(f"   Fraudes: {(df_balanced['Class']==1).sum():,} ({(df_balanced['Class']==1).mean()*100:.2f}%)")
    print(f"   Normales: {(df_balanced['Class']==0).sum():,} ({(df_balanced['Class']==0).mean()*100:.2f}%)")
    
    print(f"\n✅ LISTO PARA USO EN STREAMLIT")


if __name__ == "__main__":
    main()