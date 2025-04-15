import requests

MICROLINK_API = 'https://api.microlink.io/?url='

def validate_urls(urls):
    format_errors = []
    live_url_errors = {}
    valid_urls = []

    for i, url in enumerate(urls):
        if '.' not in url or len(url) < 5:
            format_errors.append(f"Entry {i+1} looks incomplete.")
        elif not (url.startswith("http://") or url.startswith("https://")):
            format_errors.append(f"Entry {i+1} is missing http:// or https://")
        else:
            try:
                res = requests.get(f"{MICROLINK_API}{url}")
                data = res.json()
                if data.get("status") != "success":
                    live_url_errors[url] = "URL is unreachable or invalid"
                else:
                    valid_urls.append(url)
            except:
                live_url_errors[url] = "Network error validating URL"

    return format_errors, live_url_errors, valid_urls

def get_page_titles(valid_urls):
    results = []
    for index, url in enumerate(valid_urls):
        try:
            r = requests.get(f"{MICROLINK_API}{url}")
            data = r.json()
            title = data.get("data", {}).get("title", "No title found")
        except:
            title = "Title not available"

        results.append({
            "Page": f"Page {chr(65 + index)}",
            "Title": title,
            "URL": url
        })

    return results
