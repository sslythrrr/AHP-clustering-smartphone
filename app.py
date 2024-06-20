from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from ahp import AHP
import locale

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def format_price_range(price_range):
    min_price, max_price = map(int, price_range.split('-'))
    
    if min_price == 0 and max_price == 1999999:
        min_price_formatted = "Rp0"
        max_price_formatted = "under Rp2.000.000"
        return f"{max_price_formatted}"
    
    if min_price == 2000000 and max_price == 4999999:
        min_price_formatted = "Rp2.000.000"
        max_price_formatted = "Rp4.999.999"
        return f"{max_price_formatted}"
    
    if min_price == 5000000 and max_price == 7999999:
        min_price_formatted = "Rp5.000.000"
        max_price_formatted = "Rp7.999.999"
        return f"{max_price_formatted}"
    
    if min_price == 8000000 and max_price == 10000000:
        min_price_formatted = "Rp8.000.000"
        max_price_formatted = "Rp10.000.000"
        return f"{max_price_formatted}"
    
    if min_price == 10000001 and max_price == 50000000:
        max_price_formatted = "above Rp10.000.000"
        return f"{max_price_formatted}"
        
    return f"{min_price_formatted} - {max_price_formatted}"

def get_filtered_recommendations(price_range, priority):
    df = pd.read_excel('smartphone.xlsx')

    min_price, max_price = map(int, price_range.split('-'))
    filtered_df = df[(df['Harga (RP)'] >= min_price) & (df['Harga (RP)'] <= max_price)]

    features = filtered_df[['CPU Cores', 'Speed (GHz)', 'RAM (GB)', 'ROM (GB)', 'Screen (in)', 'Camera (MP)', 'Battery (mAh)']]
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    n_clusters = 3

    if priority == 'performance':
        performance_features = features_scaled[:, [0, 1, 2, 3]]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(performance_features)
        cluster_scores = kmeans.transform(performance_features).sum(axis=1)
    elif priority == 'multimedia':
        multimedia_features = features_scaled[:, [5, 4, 3, 6]]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(multimedia_features)
        cluster_scores = kmeans.transform(multimedia_features).sum(axis=1)
    elif priority == 'allrounder':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features_scaled)
        cluster_scores = kmeans.transform(features_scaled).sum(axis=1)

    filtered_df['Cluster'] = kmeans.labels_
    filtered_df['Cluster_Score'] = cluster_scores

    best_cluster = filtered_df.groupby('Cluster')['Cluster_Score'].mean().idxmin()

    best_cluster_df = filtered_df[filtered_df['Cluster'] == best_cluster]

    best_cluster_df = best_cluster_df[(best_cluster_df['Harga (RP)'] >= min_price) & (best_cluster_df['Harga (RP)'] <= max_price)]

    best_cluster_df = best_cluster_df.drop(columns=['Cluster', 'Cluster_Score'])

    return best_cluster_df

@app.route('/recommendation', methods=['POST'])
def recommendation():
    price_range = request.form['price_range']
    priorities = request.form.getlist('priorities')

    if len(priorities) != 1:
        error = "Please select one preference."
        return render_template('index.html', error=error)

    priority = priorities[0]
    pr = price_range
    
    df = pd.read_excel('smartphone.xlsx')

    total_count = len(df)

    min_price, max_price = map(int, price_range.split('-'))
    filtered_df = df[(df['Harga (RP)'] >= min_price) & (df['Harga (RP)'] <= max_price)]

    features = filtered_df[['CPU Cores', 'Speed (GHz)', 'RAM (GB)', 'ROM (GB)', 'Screen (in)', 'Camera (MP)', 'Battery (mAh)']]
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    n_clusters = 3

    if priority == 'performance':
        performance_features = features_scaled[:, [0, 1, 2, 3]]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(performance_features)
        cluster_scores = kmeans.transform(performance_features).sum(axis=1)
    elif priority == 'multimedia':
        multimedia_features = features_scaled[:, [5, 4, 3, 6]]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(multimedia_features)
        cluster_scores = kmeans.transform(multimedia_features).sum(axis=1)
    elif priority == 'allrounder':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features_scaled)
        cluster_scores = kmeans.transform(features_scaled).sum(axis=1)

    filtered_df['Cluster'] = kmeans.labels_
    filtered_df['Cluster_Score'] = cluster_scores

    best_cluster = filtered_df.groupby('Cluster')['Cluster_Score'].mean().idxmin()

    best_cluster_df = filtered_df[filtered_df['Cluster'] == best_cluster]

    best_cluster_df = best_cluster_df[(best_cluster_df['Harga (RP)'] >= min_price) & (best_cluster_df['Harga (RP)'] <= max_price)]

    best_cluster_df = best_cluster_df.drop(columns=['Cluster_Score'])

    displayed_count = len(best_cluster_df)

    table_html = best_cluster_df.to_html(classes='table recommendation-table compact-table', index=False)
    
    formatted_price_range = format_price_range(pr)
    return render_template('recommendation.html', table_html=table_html, pricerange=price_range, pr=formatted_price_range, priorities=priority, displayed_count=displayed_count, total_count=total_count)



@app.route('/ahp', methods=['POST'])
def ahp():
    price_range = request.form['price_range']
    priority = request.form['priorities']

    best_cluster_df = get_filtered_recommendations(price_range, priority)

    features = best_cluster_df[['CPU Cores', 'Speed (GHz)', 'RAM (GB)', 'ROM (GB)', 'Screen (in)', 'Camera (MP)', 'Battery (mAh)']]
    labels = best_cluster_df['Model']

    pairwise_matrix_performance = np.array([
        [1, 1, 1, 1/3, 1/9, 1/9, 1/9],
        [1, 1, 1, 1/3, 1/9, 1/9, 1/9],
        [1, 1, 1, 1/3, 1/9, 1/9, 1/9],
        [3, 3, 3, 1, 1/7, 1/7, 1/7],
        [9, 9, 9, 7, 1, 1, 1],
        [9, 9, 9, 7, 1, 1, 1],
        [9, 9, 9, 7, 1, 1, 1]
    ])

    pairwise_matrix_multimedia = np.array([
        [1, 1, 1, 7, 9, 9, 9],
        [1, 1, 1, 7, 9, 9, 9],
        [1, 1, 1, 7, 9, 9, 9],
        [1/7, 1/7, 1/7, 1, 3, 3, 3],
        [1/9, 1/9, 1/9, 1/3, 1, 1, 1],
        [1/9, 1/9, 1/9, 1/3, 1, 1, 1],
        [1/9, 1/9, 1/9, 1/3, 1, 1, 1]
    ])

    pairwise_matrix_all_rounder = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ])

    if priority == 'performance':
        ahp = AHP(pairwise_matrix=pairwise_matrix_performance)
    elif priority == 'multimedia':
        ahp = AHP(pairwise_matrix=pairwise_matrix_multimedia)
    elif priority == 'allrounder':
        ahp = AHP(pairwise_matrix=pairwise_matrix_all_rounder)

    if ahp.consistency_ratio >= 0.1:
        return "Consistency ratio is too high. Please revise the pairwise comparison matrix."

    ahp_scores = ahp.score_alternatives(features)

    best_cluster_df['AHP_Score'] = ahp_scores

    sorted_df = best_cluster_df.sort_values(by='AHP_Score', ascending=False).head(10)

    sorted_df['Rank'] = range(1, 11)

    sorted_df = sorted_df[['Rank', 'Model', 'AHP_Score'] + [col for col in sorted_df.columns if col not in ['Rank', 'Model', 'AHP_Score']]]

    table_data = sorted_df.to_dict('records')

    chart_data = sorted_df[['Model', 'AHP_Score']].to_dict(orient='records')

    formatted_price_range = format_price_range(price_range)
    return render_template('ahp.html', table_data=table_data, chart_data=chart_data, pricerange=formatted_price_range, priorities=priority)

if __name__ == '__main__':
    app.run(debug=True)