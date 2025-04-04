
from urllib.parse import urlparse

def build_abstract_url(url: str) -> str:
    # first, we need to parse the URL
    parsed_url = urlparse(url)
    query = parsed_url.query

    # convert the query into a dictionary
    query_parts = query.split("&")
    query_dict = {}

    for part in query_parts:
        if (part == ""):
            continue
        split_parts = part.split("=")
        if (len(split_parts) == 0):
            continue

        if (len(split_parts) == 1):
            key, value = part, ""
        else:
            key = split_parts[0]
            value = "=".join(split_parts[1:])
            query_dict[key] = value

    # replace the values with placeholders
    for key in query_dict:
        query_dict[key] = f"<{key}>"

    # convert the dictionary back into a query string
    query = "&".join([f"{key}={value}" for key, value in query_dict.items()]) if len(query_dict.items()) > 0 else ""
    fragment = "#<fragment>" if parsed_url.fragment else ""

    # rebuild the URL
    abstract_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}{query}{fragment}"
    return abstract_url

_to_replace = [
    ("http://ec2-3-16-13-240.us-east-2.compute.amazonaws.com:3000", "https://osm.org"),
    ("http://ec2-3-16-13-240.us-east-2.compute.amazonaws.com:7770", "https://shopping.com"),
    ("http://ec2-3-16-13-240.us-east-2.compute.amazonaws.com:7780", "https://shopping.com"),
    ("http://ec2-3-16-13-240.us-east-2.compute.amazonaws.com:8023", "https://vcs.com"),
    ("http://ec2-3-16-13-240.us-east-2.compute.amazonaws.com:9980", "https://marketplace.com"),
    ("http://ec2-3-16-13-240.us-east-2.compute.amazonaws.com:9999", "https://social-forum.com"),
    ("https://dev275528.service-now.com", "https://servicenow.test"),
]

def replace_urls(url, to_replace = _to_replace):
    for old, new in to_replace:
        url = url.replace(old, new)
    return url

