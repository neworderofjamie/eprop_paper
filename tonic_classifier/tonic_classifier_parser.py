from argparse import ArgumentParser

def parse_arguments(parent_parser=None, description=None):
    # Build command line parse
    if parent_parser is None:
        parser = ArgumentParser(description=description)
    else:
        parser = ArgumentParser(description=description, parents=[parent_parser])
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--crop-time", type=float, default=None)
    parser.add_argument("--feedforward", action="store_true")
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--deep-r", action="store_true")
    parser.add_argument("--num-recurrent-alif", type=int, default=256)
    parser.add_argument("--num-recurrent-lif", type=int, default=0)
    parser.add_argument("--num-hidden", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--learning-rate-decay", type=float, default=1.0)
    parser.add_argument("--learning-rate-decay-epochs", type=int, default=0)
    parser.add_argument("--log-latency-threshold", type=int, default=51)
    parser.add_argument("--log-latency-tau", type=float, default=20.0)
    parser.add_argument("--regularizer-strength", type=float, default=0.001)
    parser.add_argument("--l1-regularizer-strength", type=float, default=0.01)
    parser.add_argument("--input-recurrent-sparsity", type=float, default=1.0)
    parser.add_argument("--input-recurrent-max-row-length", type=int, default=None)
    parser.add_argument("--input-recurrent-inh-sparsity", type=float, default=None)
    parser.add_argument("--recurrent-recurrent-max-row-length", type=int, default=None)
    parser.add_argument("--recurrent-recurrent-sparsity", type=float, default=1.0)
    parser.add_argument("--recurrent-excitatory-fraction", type=float, default=0.8)
    parser.add_argument("--dataset", choices=["smnist", "shd", "dvs_gesture", "mnist"], required=True)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reset-neurons", action="store_true")
    args = parser.parse_args()

    # Determine output directory name and create if it doesn't exist
    name_suffix = "%u%s%s%s%s%s%s%s" % (args.num_recurrent_alif, 
                                  "_%u" % args.num_recurrent_lif if args.num_recurrent_lif > 0 else "",
                                  "_%u" % args.num_hidden if args.num_hidden > 1 else "",
                                  "_feedforward" if args.feedforward else "", 
                                  "_no_bias" if args.no_bias else "", 
                                  "_%.1f" % args.dt if args.dt != 1.0 else "",
                                  "_%.1f" % args.crop_time if args.crop_time is not None else "",
                                  args.suffix)
    output_directory = "%s_%s" % (args.dataset, name_suffix)

    return name_suffix, output_directory, args
