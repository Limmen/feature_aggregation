from large_pomdp_parser import parse_pomdp_data, save_model, sample_next_state_and_obs

if __name__ == '__main__':
    parsed_model = parse_pomdp_data("RockSample_4_4.pomdp")
    save_model(parsed_model, "rocksample_4_4.pkl")