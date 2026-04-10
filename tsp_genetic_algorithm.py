import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from typing import List, Tuple

# Rastgelelik için seed (tekrarlanabilirlik)
random.seed(42)
np.random.seed(42)

# ===================== HİPERPARAMETRELER =====================
POPULATION_SIZE = 100       # Popülasyon boyutu
NUM_GENERATIONS = 200       # Nesil (iterasyon) sayısı
ELITE_SIZE = 10             # Elitizm: korunacak en iyi birey sayısı
MUTATION_RATE = 0.15        # Mutasyon oranı
CROSSOVER_RATE = 0.85       # Çaprazlama oranı
TOURNAMENT_SIZE = 5         # Tournament seçiminde yarışacak birey sayısı

# ===================== VERİ YÜKLEME =====================
def load_distance_matrix(filepath: str) -> np.ndarray:
    """Excel'den mesafe matrisini yükler."""
    df = pd.read_excel(filepath, header=None)
    distance_matrix = df.iloc[1:22, 1:22].values.astype(float)
    return distance_matrix

# ===================== FITNESS FONKSİYONU =====================
def calculate_total_distance(route: List[int], dist_matrix: np.ndarray) -> float:
    """Bir rotanın toplam mesafesini hesaplar."""
    total = 0
    full_route = [0] + route + [0]
    for i in range(len(full_route) - 1):
        total += dist_matrix[full_route[i], full_route[i + 1]]
    return total

def fitness(route: List[int], dist_matrix: np.ndarray) -> float:
    """Fitness değeri: mesafe ne kadar düşükse fitness o kadar yüksek."""
    return 1 / calculate_total_distance(route, dist_matrix)

# ===================== BAŞLANGIÇ POPÜLASYONU =====================
def create_initial_population(pop_size: int, num_cities: int) -> List[List[int]]:
    """Rastgele başlangıç popülasyonu oluşturur."""
    population = []
    cities = list(range(1, num_cities))
    for _ in range(pop_size):
        individual = cities.copy()
        random.shuffle(individual)
        population.append(individual)
    return population

# ===================== SEÇİM YÖNTEMİ: TOURNAMENT =====================
def tournament_selection(population: List[List[int]], dist_matrix: np.ndarray, 
                         tournament_size: int) -> List[int]:
    """Tournament seçimi: rastgele k birey seç, en iyisini döndür."""
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda x: calculate_total_distance(x, dist_matrix))
    return tournament[0]

# ===================== ÇAPRAZLAMA: ORDER CROSSOVER (OX) =====================
def order_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """Order Crossover (OX): TSP için uygun çaprazlama yöntemi."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child1 = [None] * size
    child1[start:end] = parent1[start:end]
    remaining = [gene for gene in parent2 if gene not in child1]
    idx = 0
    for i in range(size):
        if child1[i] is None:
            child1[i] = remaining[idx]
            idx += 1
    
    child2 = [None] * size
    child2[start:end] = parent2[start:end]
    remaining = [gene for gene in parent1 if gene not in child2]
    idx = 0
    for i in range(size):
        if child2[i] is None:
            child2[i] = remaining[idx]
            idx += 1
    
    return child1, child2

# ===================== MUTASYON: SWAP + INVERSION =====================
def swap_mutation(route: List[int]) -> List[int]:
    """Swap mutasyonu: iki rastgele şehri yer değiştirir."""
    mutated = route.copy()
    idx1, idx2 = random.sample(range(len(mutated)), 2)
    mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    return mutated

def inversion_mutation(route: List[int]) -> List[int]:
    """Inversion mutasyonu: bir segmenti tersine çevirir."""
    mutated = route.copy()
    start, end = sorted(random.sample(range(len(mutated)), 2))
    mutated[start:end] = mutated[start:end][::-1]
    return mutated

def mutate(route: List[int], mutation_rate: float) -> List[int]:
    """Mutasyon uygula: %50 swap, %50 inversion."""
    if random.random() < mutation_rate:
        if random.random() < 0.5:
            return swap_mutation(route)
        else:
            return inversion_mutation(route)
    return route

# ===================== YENİ NESİL OLUŞTURMA =====================
def create_new_generation(population: List[List[int]], dist_matrix: np.ndarray,
                          elite_size: int, mutation_rate: float, 
                          crossover_rate: float, tournament_size: int) -> List[List[int]]:
    """Yeni nesil oluşturur."""
    new_population = []
    sorted_pop = sorted(population, key=lambda x: calculate_total_distance(x, dist_matrix))
    new_population.extend(sorted_pop[:elite_size])
    
    while len(new_population) < len(population):
        parent1 = tournament_selection(population, dist_matrix, tournament_size)
        parent2 = tournament_selection(population, dist_matrix, tournament_size)
        
        if random.random() < crossover_rate:
            child1, child2 = order_crossover(parent1, parent2)
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        
        new_population.append(child1)
        if len(new_population) < len(population):
            new_population.append(child2)
    
    return new_population

# ===================== GRAFİK ÇİZİMİ =====================
def plot_route(route: List[int], dist_matrix: np.ndarray, generation: int, distance: float):
    """En iyi rotayı grafik üzerinde çizdirir."""
    num_cities = len(dist_matrix)
    symmetric_matrix = (dist_matrix + dist_matrix.T) / 2
    
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto', n_init=4)
    coords = mds.fit_transform(symmetric_matrix)
    
    full_route = [0] + route + [0]
    
    plt.figure(figsize=(12, 8))
    
    for i in range(len(full_route) - 1):
        x_vals = [coords[full_route[i], 0], coords[full_route[i + 1], 0]]
        y_vals = [coords[full_route[i], 1], coords[full_route[i + 1], 1]]
        plt.plot(x_vals, y_vals, 'b-', linewidth=1.5, alpha=0.7)
        
        mid_x = (x_vals[0] + x_vals[1]) / 2
        mid_y = (y_vals[0] + y_vals[1]) / 2
        dx = x_vals[1] - x_vals[0]
        dy = y_vals[1] - y_vals[0]
        plt.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), 
                     xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                     arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    for i in range(num_cities):
        if i == 0:
            plt.scatter(coords[i, 0], coords[i, 1], c='red', s=200, zorder=5, edgecolors='black')
            plt.annotate(f'{i}\n(Baslangic)', (coords[i, 0], coords[i, 1]), 
                        textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
        else:
            plt.scatter(coords[i, 0], coords[i, 1], c='lightgreen', s=150, zorder=5, edgecolors='black')
            plt.annotate(str(i), (coords[i, 0], coords[i, 1]), 
                        textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)
    
    plt.title(f'TSP - Genetik Algoritma Cozumu\nEn Iyi Mesafe: {distance:.2f} | Nesil: {generation}', fontsize=14)
    plt.xlabel('X Koordinati')
    plt.ylabel('Y Koordinati')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsp_best_route.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nGrafik 'tsp_best_route.png' olarak kaydedildi.")

def plot_convergence(history: List[float]):
    """Yakınsama grafiği çizer."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, 'b-', linewidth=2)
    plt.xlabel('Nesil (Iterasyon)', fontsize=12)
    plt.ylabel('En Iyi Mesafe', fontsize=12)
    plt.title('Genetik Algoritma Yakinsama Grafigi', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsp_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Yakinsama grafigi 'tsp_convergence.png' olarak kaydedildi.")

# ===================== SONUÇLARI DOSYAYA KAYDET =====================
def save_results_to_file(iteration_results: List[dict], best_route: List[int], 
                         best_distance: float, filename: str = 'iteration_results.txt'):
    """İterasyon sonuçlarını dosyaya kaydeder."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("GENETIK ALGORITMA - TSP SONUCLARI\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("HIPERPARAMETRELER:\n")
        f.write(f"  Populasyon Boyutu: {POPULATION_SIZE}\n")
        f.write(f"  Nesil Sayisi: {NUM_GENERATIONS}\n")
        f.write(f"  Elit Birey Sayisi: {ELITE_SIZE}\n")
        f.write(f"  Caprazlama Orani: {CROSSOVER_RATE}\n")
        f.write(f"  Mutasyon Orani: {MUTATION_RATE}\n")
        f.write(f"  Tournament Boyutu: {TOURNAMENT_SIZE}\n\n")
        
        f.write("KULLANILAN YONTEMLER:\n")
        f.write("  Secim Yontemi: Tournament Selection\n")
        f.write("  Caprazlama Yontemi: Order Crossover (OX)\n")
        f.write("  Mutasyon Yontemi: Swap Mutation + Inversion Mutation\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("ITERASYON SONUCLARI:\n")
        f.write("=" * 70 + "\n\n")
        
        for result in iteration_results:
            f.write(f"Iteration {result['iteration']:3d} -> Best Distance: {result['best_distance']:.2f}\n")
            f.write(f"Best Route: {result['best_route']}\n")
            f.write("-" * 70 + "\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("EN IYI SONUC:\n")
        f.write("=" * 70 + "\n")
        f.write(f"En Iyi Mesafe: {best_distance:.2f}\n")
        f.write(f"En Iyi Rota: {[0] + best_route + [0]}\n")
    
    print(f"\nSonuclar '{filename}' dosyasina kaydedildi.")

# ===================== ANA GENETİK ALGORİTMA =====================
def genetic_algorithm(dist_matrix: np.ndarray) -> Tuple[List[int], float, List[dict], List[float]]:
    """Ana genetik algoritma fonksiyonu."""
    num_cities = len(dist_matrix)
    population = create_initial_population(POPULATION_SIZE, num_cities)
    
    best_route = None
    best_distance = float('inf')
    history = []
    iteration_results = []
    
    print("=" * 70)
    print("GENETIK ALGORITMA - GEZGIN SATICI PROBLEMI (TSP)")
    print("=" * 70)
    print(f"Populasyon Boyutu: {POPULATION_SIZE}")
    print(f"Nesil Sayisi: {NUM_GENERATIONS}")
    print(f"Elit Birey Sayisi: {ELITE_SIZE}")
    print(f"Caprazlama Orani: {CROSSOVER_RATE}")
    print(f"Mutasyon Orani: {MUTATION_RATE}")
    print(f"Tournament Boyutu: {TOURNAMENT_SIZE}")
    print("=" * 70)
    print()
    
    for generation in range(1, NUM_GENERATIONS + 1):
        for individual in population:
            distance = calculate_total_distance(individual, dist_matrix)
            if distance < best_distance:
                best_distance = distance
                best_route = individual.copy()
        
        full_route = [0] + best_route + [0]
        history.append(best_distance)
        iteration_results.append({
            'iteration': generation,
            'best_distance': best_distance,
            'best_route': full_route.copy()
        })
        
        print(f"Iteration {generation:3d} -> Best Distance: {best_distance:.2f}")
        print(f"Best Route: {full_route}")
        print("-" * 70)
        
        population = create_new_generation(
            population, dist_matrix, ELITE_SIZE, 
            MUTATION_RATE, CROSSOVER_RATE, TOURNAMENT_SIZE
        )
    
    return best_route, best_distance, iteration_results, history

# ===================== ANA PROGRAM =====================
if __name__ == "__main__":
    # Mesafe matrisini yükle
    dist_matrix = load_distance_matrix('Distance_matrix.xlsx')
    
    print(f"Mesafe matrisi yuklendi: {dist_matrix.shape[0]} dugum\n")
    
    # Genetik algoritmayı çalıştır
    best_route, best_distance, iteration_results, history = genetic_algorithm(dist_matrix)
    
    # Sonuçları göster
    print("\n" + "=" * 70)
    print("SONUC")
    print("=" * 70)
    print(f"En Iyi Mesafe: {best_distance:.2f}")
    print(f"En Iyi Rota: {[0] + best_route + [0]}")
    
    # Grafikleri çiz
    plot_route(best_route, dist_matrix, NUM_GENERATIONS, best_distance)
    plot_convergence(history)
    
    # Sonuçları dosyaya kaydet
    save_results_to_file(iteration_results, best_route, best_distance)
    
    print("\n" + "=" * 70)
    print("TAMAMLANDI!")
    print("Olusturulan dosyalar:")
    print("  - tsp_best_route.png (Rota grafigi)")
    print("  - tsp_convergence.png (Yakinsama grafigi)")
    print("  - iteration_results.txt (Tum iterasyon sonuclari)")
    print("=" * 70)
