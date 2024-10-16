from huggingface_hub import snapshot_download
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id", 
        type=str,   
        required=True, 
        help="A user or an organization name and a repo name separated by a / such as 'Qwen/Qwen-72B-Chat'"
    )
    parser.add_argument(
        "--repo_type", 
        type=str, 
        default=None, 
        help="Set to \"dataset\" or \"space\" if downloading from a dataset or space, None or \"model\" if downloading from a model. Default is None."
    )
    parser.add_argument(
        "--revision", 
        type=str, 
        default=None, 
        help="The revision of the repo to download. Can be a branch name, a tag name, or a commit id."
    )
    parser.add_argument(
        "--proxy", 
        type=str, 
        default=None,
        help="Proxy address, e.g. http://127.0.0.1:7890. If 'auto' is set, 'http://127.0.0.1:7890' is used. Note: Remember to use http instead of https for the proxy value."
    )
    parser.add_argument( 
        "--parent_dir",
        type=str, 
        default=None, 
        help="the parent directory of model folder downloaded from hugging face."
    )
    args = parser.parse_args()

    if args.proxy is not None:
        if args.proxy == "auto":
            args.proxy = "http://127.0.0.1:7890" 
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy

    local_dir = args.repo_id.split('/')[1]
    if args.parent_dir is not None:
        local_dir = os.path.join(args.parent_dir, local_dir)

    # download the model
    snapshot_download(
        repo_id=args.repo_id, 
        local_dir=local_dir, 
        repo_type=args.repo_type, 
        revision=args.revision
    )
