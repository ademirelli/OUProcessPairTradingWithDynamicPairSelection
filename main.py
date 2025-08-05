from pair_discovery.discover_pairs import discover_top_pairs

if __name__ == "__main__":
    print("Running pair discovery...\n")
    top_pairs_df = discover_top_pairs()
    print(top_pairs_df[["pair", "correlation", "cointegration_p", "spread_var", "score"]]) 
