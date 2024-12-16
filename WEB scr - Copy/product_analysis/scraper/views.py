from django.shortcuts import render
from .utils import scrape_ebay, scrape_snapdeal, scrape_amazon, scrape_ajio
from .models import Product, AmazonProduct, eBayProduct, SnapdealProduct, AjioProduct
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Function to rank products using clustering and calculate a predicted score
def rank_products_ml(products):
    df = pd.DataFrame(products)

    # Clean the 'price' column
    df['price'] = df['price'].replace({'\u20b9': '', ',': ''}, regex=True).replace('', np.nan)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')  # Convert to numeric, invalid entries become NaN

    # Drop rows where 'price' is NaN or non-numeric
    df.dropna(subset=['price'], inplace=True)

    # Clean the 'rating' column
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')  # Convert to numeric, invalid entries become NaN
    df['rating'].fillna(0, inplace=True)  # Replace NaN ratings with 0

    # Normalize 'price' and 'rating'
    if not df.empty:  # Check if DataFrame is not empty
        scaler = MinMaxScaler()
        df[['normalized_price', 'normalized_rating']] = scaler.fit_transform(df[['price', 'rating']])
    else:
        df['normalized_price'] = df['normalized_rating'] = []

    # Clustering with K-Means (only if there's enough data)
    if len(df) >= 3:  # Ensure at least 3 data points for clustering
        X = df[['normalized_price', 'normalized_rating']].values
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)
        # Compute distances to centroid
        df['distance_to_centroid'] = np.linalg.norm(X - kmeans.cluster_centers_[df['cluster']], axis=1)
        # Calculate predicted scores
        df['predicted_score'] = (1 / (1 + df['distance_to_centroid'])) * df['rating']
        # Sort by predicted score
        df = df.sort_values(by='predicted_score', ascending=False)
    else:
        df['cluster'] = df['distance_to_centroid'] = df['predicted_score'] = []

    return df


# Function to predict future prices using linear regression and draw graphs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64

def visualize_price_data(prices, labels=None, future_steps=10):
    # Create a DataFrame for the price data
    df = pd.DataFrame(prices, columns=['price'])

    # Use provided labels for X-axis, or default to indices
    if labels:
        df['label'] = labels
    else:
        df['label'] = range(len(df))

    # Train a Linear Regression model
    model = LinearRegression()
    X = df['label'].values.reshape(-1, 1)  # Feature (Product index)
    y = df['price'].values  # Target (Price)
    model.fit(X, y)

    # Predict future prices
    future_labels = np.arange(len(df), len(df) + future_steps).reshape(-1, 1)
    future_prices = model.predict(future_labels)

    # Plot the actual prices
    plt.figure(figsize=(10, 6))
    plt.plot(df['label'], df['price'], label='Prices', marker='o' , color='blue')
    
    

    # Plot the predicted future prices
    plt.plot(future_labels, future_prices, label=f'Predicted Future Prices ({future_steps} steps)',linestyle='--', color='red')

    plt.xlabel('Products' if labels else 'Index')
    plt.ylabel('Price')
    plt.title('Price Visualization with Future Price Prediction')
    plt.xticks(rotation=45)  # Rotate X-axis labels for better readability
    plt.legend()
    
    

    # Save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encode the plot as Base64
    graph = base64.b64encode(image_png).decode('utf-8')

    return graph, future_prices


# View function to scrape products and predict prices
def scrape_products(request):
    query = request.GET.get('query', '')  # Default to empty if no query is provided

    if query:
        # Scrape all the products for each platform
     #   amazon_products = scrape_amazon(query)
        ajio_products = scrape_ajio(query)
        ebay_products = scrape_ebay(query)
        snapdeal_products = scrape_snapdeal(query)

        # Rank the products using ML
     #   top_3_amazon = rank_products_ml(amazon_products).head(3)[['title', 'price', 'rating', 'link']]
        top_3_ajio = rank_products_ml(ajio_products).head(3)[['title', 'price', 'rating', 'link']]
        top_3_ebay = rank_products_ml(ebay_products).head(3)[['title', 'price', 'rating', 'link']]
        top_3_snapdeal = rank_products_ml(snapdeal_products).head(3)[['title', 'price', 'rating', 'link']] if snapdeal_products else []

        # Extract prices for prediction
    #    amazon_prices = [float(p['price'].replace('₹', '').replace(',', '').replace('No Price Available', '').strip()) for p in amazon_products if p.get('price') and p['price'] != 'No Price Available']
        ajio_prices = [float(p['price'].replace('₹', '').replace(',', '').replace('No Price Available', '').strip()) for p in ajio_products if p.get('price') and p['price'] != 'No Price Available']
        
        # Extract prices for prediction (ensure all prices are strings before applying replace)
        ebay_prices = [
             float(str(p['price']).replace('₹', '').replace(',', '').replace('No Price Available', '').strip()) 
                for p in ebay_products 
                    if p.get('price') and p['price'] != 'No Price Available'
]

        snapdeal_prices = [float(p['price'].replace('₹', '').replace(',', '').replace('No Price Available', '').strip()) for p in snapdeal_products if p.get('price') and p['price'] != 'No Price Available']

        # Call the function to predict future prices and generate the graph
     #   graph_amazon, future_amazon_prices = visualize_price_data(amazon_prices)
        graph_ajio, future_ajio_prices = visualize_price_data(ajio_prices)
        graph_ebay, future_ebay_prices = visualize_price_data(ebay_prices)
        graph_snapdeal, future_snapdeal_prices = visualize_price_data(snapdeal_prices)
        
        

        # Prepare the context to be passed to the template
        context = {
          #  'amazon_products': amazon_products,
            'ebay_products': ebay_products,
            'snapdeal_products': snapdeal_products,
            'ajio_products': ajio_products,
            'top_3_snapdeal': top_3_snapdeal.to_dict(orient='records') if not top_3_snapdeal.empty else [],
            'top_3_ajio': top_3_ajio.to_dict(orient='records'),
          #  'top_3_amazon': top_3_amazon.to_dict(orient='records'),
            'top_3_ebay': top_3_ebay.to_dict(orient='records'),
           #  'graph_amazon': graph_amazon,
            'graph_ajio': graph_ajio,
            'graph_ebay': graph_ebay,
            'graph_snapdeal': graph_snapdeal,
          #  'future_amazon_prices': future_amazon_prices.tolist(),  # Convert numpy array to list for template
            'future_ajio_prices': future_ajio_prices.tolist(),
            'future_ebay_prices': future_ebay_prices.tolist(),
            'future_snapdeal_prices': future_snapdeal_prices.tolist(),
        }

        # Return the rendered page with all relevant product data
        return render(request, 'scraper/product_list.html', context)

    # Return an error message if no query is provided
    return render(request, 'scraper/product_list.html', {'message': 'No query provided'})
