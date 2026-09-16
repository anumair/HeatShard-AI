"""One-time ETL: pull the Olist Brazilian E-commerce dataset (via kagglehub,
needs a Kaggle API token) and prepare a compact real-product-id record set
for the simulator's key space.

Olist has no inventory/stock table, so the third leg of our product/
reviews/inventory co-access grouping is repurposed as "order" (order
volume + freight -- the closest real analogue to operational/stock
activity available in this dataset). Traffic shape itself stays fully
synthetic (Zipfian + flash-sale injection, unchanged) -- this script only
supplies a *real* catalog of record ids and payloads to draw that
synthetic traffic over.

Output: data/olist_records.json (gitignored -- regenerate locally with
your own Kaggle credentials; not redistributed since Olist's data is
CC BY-NC-SA licensed).
"""

import argparse
import json
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "olist_records.json"


def build(num_products: int, min_orders: int, out_path: Path):
    import kagglehub
    import pandas as pd

    cache_dir = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    print(f"dataset cached at {cache_dir}")

    products = pd.read_csv(f"{cache_dir}/olist_products_dataset.csv")
    items = pd.read_csv(f"{cache_dir}/olist_order_items_dataset.csv")
    reviews = pd.read_csv(f"{cache_dir}/olist_order_reviews_dataset.csv")
    orders = pd.read_csv(f"{cache_dir}/olist_orders_dataset.csv")[["order_id"]]

    order_stats = (
        items.groupby("product_id")
        .agg(order_count=("order_id", "size"), avg_price=("price", "mean"), avg_freight=("freight_value", "mean"))
        .reset_index()
    )
    order_stats = order_stats[order_stats["order_count"] >= min_orders]
    order_stats = order_stats.sort_values("order_count", ascending=False).head(num_products)

    # order_items -> order_id -> review_score (reviews key off order_id, not product_id directly)
    item_reviews = items[["order_id", "product_id"]].merge(reviews[["order_id", "review_score"]], on="order_id")
    review_stats = (
        item_reviews.groupby("product_id")
        .agg(avg_review_score=("review_score", "mean"), review_count=("review_score", "size"))
        .reset_index()
    )

    merged = (
        order_stats.merge(products, on="product_id", how="left")
        .merge(review_stats, on="product_id", how="left")
    )
    merged["avg_review_score"] = merged["avg_review_score"].fillna(0.0)
    merged["review_count"] = merged["review_count"].fillna(0).astype(int)

    product_ids = merged["product_id"].tolist()
    records = {}
    for row in merged.itertuples():
        records[row.product_id] = {
            "product": {
                "category": row.product_category_name if isinstance(row.product_category_name, str) else "unknown",
                "avg_price": round(float(row.avg_price), 2),
                "weight_g": None if pd.isna(row.product_weight_g) else float(row.product_weight_g),
            },
            "review": {
                "avg_score": round(float(row.avg_review_score), 2),
                "review_count": int(row.review_count),
            },
            "order": {
                "order_count": int(row.order_count),
                "avg_freight": round(float(row.avg_freight), 2),
            },
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"product_ids": product_ids, "records": records}, f, indent=2)

    print(f"wrote {len(product_ids)} real products to {out_path}")
    print(f"order_count range: {merged['order_count'].min()}..{merged['order_count'].max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a real-product-id key space from the Olist dataset")
    parser.add_argument("--num-products", type=int, default=200)
    parser.add_argument("--min-orders", type=int, default=5, help="drop products with fewer real orders than this")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    build(args.num_products, args.min_orders, Path(args.out))
