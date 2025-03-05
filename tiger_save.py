from large_pomdp_parser import parse_pomdp_data, save_model

if __name__ == '__main__':
    parsed_model = parse_pomdp_data("tiger.pomdp")
    save_model(parsed_model, "tiger_model.pkl")