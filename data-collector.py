#!/usr/bin/env python3

# for artstation: https://www.artstation.com/search/projects.json?direction=desc&order=published_at&page=1&q=wallpaper&show_pro_first=true

import requests
import os, uuid, time, json, re, zipfile, argparse
import logging, logging.handlers, coloredlogs
from multiprocessing import Pool, cpu_count, Queue, Manager
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PIL import Image
import threading
import pdb

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
FORMAT = '%(asctime)-15s %(pid)-5s %(threadName)-10s %(worker)-8s %(levelname)-8s %(message)s'
log_formatter = logging.Formatter(FORMAT)
# log_formatter.
logging.basicConfig(format=FORMAT, level=logging.INFO, filename="data-collector.log")
log = logging.getLogger(__name__)
coloredlogs.DEFAULT_LEVEL_STYLES["debug"]["color"] = "white"
coloredlogs.DEFAULT_LEVEL_STYLES["spam"]["color"] = "white"
coloredlogs.DEFAULT_FIELD_STYLES["levelname"] = {}
coloredlogs.DEFAULT_FIELD_STYLES["asctime"]["color"] = "white"
coloredlogs.DEFAULT_FIELD_STYLES["pid"] = {"color":"cyan"}
coloredlogs.DEFAULT_FIELD_STYLES["threadName"] = {"color":"yellow"}
coloredlogs.DEFAULT_FIELD_STYLES["worker"] = {"color":"magenta"}
coloredlogs.install(level='INFO', fmt=FORMAT)

# log_handler_file = logging.handlers.RotatingFileHandler("data-collector.log", backupCount=5)
# log_handler_file.setLevel(logging.DEBUG)
# log_handler_file.setFormatter(FORMAT)
# log.addHandler(log_handler_file)

# log_handler_stream = logging.StreamHandler()
# log_handler_stream.setLevel(logging.INFO)
# log_handler_stream.setFormatter(FORMAT)
# log.addHandler(log_handler_stream)

minimum_resolution = (1440, 900)
max_crawl_queue = -1
max_download_queue = 10000
download_location = Path("./data/raw")

blacklist_hostname = ["help.deviantart.net", "about.deviantart.net",
	"comments.deviantart.net", "comments.deviantart.com", "shop.deviantart.net", "facebook.com",
	"twitter.com", "gravatar.com", "youtube.com", "s.ytimg.com",
	"tvfiles.alphacoders.com", "vidfiles.alphacoders.com",
	"giffiles.alphacoders.com", "gamefiles.alphacoders.com",
	"paypalobjects.com", "commons.wikimedia.org", "play.google.com", "abc.xyz",
	"doubleclick.com", "s.w.org", "wordpress.org",
	"gweb-earth.storage.googleapis.com", "dl.google.com", "allo.google.com",
	"townnews.com", "enterprise.microsoft.com", "news.microsoft.com", "npmjs.com", "privacy.google.com"]
blacklist_tags = ["female", "anime", "minecraft", "comics", "movie", "women"]

queued_urls = []

def hasProcessedUrl(url):
	# return any([x == url for x in queued_urls])
	return url in queued_urls

def isBlacklisted(url):
	uri = urlparse(url.replace("www.", ""))
	log.debug("{} isBlacklisted: {}".format(uri.hostname, uri.hostname in blacklist_hostname), extra=log_extra)
	return uri.hostname in blacklist_hostname

def areTagsBlacklisted(tags):
	return any([t.lower() == b.lower() for t in tags for b in blacklist_tags])

def add_crawl(url):
	if url == "https://www.artstation.com/":
		log.warning("skiping because index of artstation {}".format(url), extra=log_extra)
		return

	uri = urlparse(url)
	if "wallpaperscraft" in uri.hostname:
		log.debug("wallpaperscraft special filtering", extra=log_extra)
		regex_resolution = re.compile("[0-9]+x[0-9]+")
		for match in regex_resolution.finditer(uri.path):
			width, height = tuple(match.group().split("x"))
			width, height = int(width), int(height)
			if width < minimum_resolution[0] and height < minimum_resolution[1]:
				log.warning("skipping because wallpaperscraft special filtering failed {}".format(url), extra=log_extra)
				return


	if isBlacklisted(url):
		log.debug("skiping because hostname is blacklisted {}".format(url), extra=log_extra)
		return

	if hasProcessedUrl(url):
		log.debug("skipping because already queued {}".format(url), extra=log_extra)
		return

	# if "big.php?i=72092" in url:
	# 	pdb.set_trace()

	log.info("adding crawl {}".format(url), extra=log_extra)
	if crawl_queue.qsize() < max_crawl_queue or max_crawl_queue == -1:
		crawl_queue.put(url)
		queued_urls.append(url)
	else:
		log.error("queue is full, can't queue {}".format(url), extra=log_extra)

def add_download(url):
	if isBlacklisted(url):
		log.debug("skiping because hostname is blacklisted {}".format(url), extra=log_extra)
		return

	uri = urlparse(url)

	if "alphacoders" in uri.hostname and "thumb" in url:
		log.debug("skipping because alphacoders thumb {}".format(url), extra=log_extra)
		return

	if hasProcessedUrl(url):
		log.debug("skipping because already queued {}".format(url), extra=log_extra)
		return

	if any([uri.path.lower().endswith(ext) for ext in [".svg", ".ico"]]):
		log.debug("skipping because svg or ico {}".format(href), extra=log_extra)
		return

	log.debug("adding download {}".format(url), extra=log_extra)
	if download_queue.qsize() < max_download_queue or max_download_queue == -1:
		download_queue.put(url)
		queued_urls.append(url)
	else:
		log.error("queue is full, can't queue {}".format(url), extra=log_extra)

def crawl_page(url):
	assert isinstance(url, str) and len(url) > 0
	log_extra = {"pid":os.getpid(), "worker":"CRAWL"}

	log.info(url, extra=log_extra)
	response = requests.get(url, timeout=10)
	if response.status_code == 200:
		log.info("success ({})".format(response.status_code), extra=log_extra)

		if "application/json" in response.headers['content-type']:
			parse_json(url, response.text)
		elif "text/html" in response.headers['content-type']:
			parse_html(url, response.text)

		uri = urlparse(url)
		if len(uri.query) > 0:
			for keyvalue in uri.query.split("&"):
				key, value = tuple(keyvalue.split("="))
				if key == "page":
					new_value = int(value) + 1
					next_page_url = uri.geturl().replace("page={}".format(value), "page={}".format(new_value))
					if not hasProcessedUrl(next_page_url):
						add_crawl(next_page_url)
					else:
						log.debug("skipping because next page has already been queued {}".format(next_page_url), extra=log_extra)
					break
	else:
		log.error("failed ({}) {}".format(response.status_code, response.url), extra=log_extra)

def parse_html(url, html):
	log_extra = {"pid":os.getpid(), "worker":"CRAWL"}

	uri = urlparse(url)
	soup = BeautifulSoup(html, 'html.parser')
	if "alphacoders" in uri.hostname and not filter_alphacoders(url, html):
		log.debug("skipping because filter_alphacoders check did not pass {}".format(url), extra=log_extra)
		return

	if "wallpaperscraft" in uri.hostname:
		if "wallpaper" in uri.path:
			tags = get_tags_wallpaperscraft(html)
			if areTagsBlacklisted(tags):
				log.debug("skipping because wallpaperscraft tags check did not pass {}".format(url), extra=log_extra)
				return
			add_download(get_highest_resolution_wallpaperscraft(html))
			return

	for hyperlink in soup.find_all(name="a"):
		if not hyperlink.get("href") or hyperlink.get("href").startswith("#"):
			continue
		href = urljoin(url, hyperlink.get("href"))
		if href and href != url:
			# queued_urls.append(href)
			uri = urlparse(href)
			# if hasProcessedUrl(url):
			# 	log.debug("skipping because already queued {}".format(href), extra=log_extra)
			# 	continue
			if hyperlink.parent.name in ["li", "form"]:
				# print(hyperlink.parent.parent["class"])
				# os._exit()
				if hyperlink.parent.name == "form":
					log.warning("skipping because parent is <{}> {}".format(hyperlink.parent.name, href), extra=log_extra)
					continue

				try:
					if "navbar-nav" in hyperlink.parent.parent["class"]:
						log.warning("skipping because parent is <{}> and in ul.navbar-nav {}".format(hyperlink.parent.name, href), extra=log_extra)
						continue
					elif "dropdown-menu" in hyperlink.parent.parent["class"]:
						log.warning("skipping because parent is <{}> and in ul.dropdown-menu {}".format(hyperlink.parent.name, href), extra=log_extra)
						continue
				except KeyError as e:
					log.debug("KeyError when checking for dropdown or navbar", extra=log_extra)
			if "forgot password" in hyperlink.text.lower():
				log.warning("skipping because detected forgot password link {}".format(href), extra=log_extra)
				continue
			if any([uri.path.lower().endswith(ext) for ext in [".svg", ".ico"]]):
				log.debug("skipping because svg or ico {}".format(href), extra=log_extra)
				continue
			if any([uri.path.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".zip"]]):
				add_download(href)
			else:
				add_crawl(href)

	for img_tag in soup.find_all("img"):
		src = urljoin(url, img_tag.get("src"))

		if img_tag.parent.name == "a":
			log.warning("skipping because img_tag parent is <a> {}".format(src), extra=log_extra)
			continue

		if any([uri.path.lower().endswith(ext) for ext in [".svg", ".ico"]]):
			log.debug("skipping because svg or ico {}".format(href), extra=log_extra)
			continue
		if not hasProcessedUrl(src):
			add_download(str(src))
			# queued_urls.append(str(src))
		else:
			log.debug("skipping because already queued {}".format(href), extra=log_extra)

	for meta_tag in soup.find_all("meta"):
		if meta_tag.get("name") == "image":
			src = urljoin(url, meta_tag.get("content"))
			uri = urlparse(src)
			if any([uri.path.lower().endswith(ext) for ext in [".svg", ".ico"]]):
				log.debug("skipping because svg or ico {}".format(href), extra=log_extra)
				continue
			if not hasProcessedUrl(src):
				# log.debug("adding download {}".format(src), extra=log_extra)
				add_download(str(src))
				# queued_urls.append(str(src))
			else:
				log.debug("skipping because meta tag is not image {}".format(href), extra=log_extra)

def filter_alphacoders(url, html):
	"""
	Filter unwanted images from alphacoders by checking the tags and the hostname.

	returns True if the image tags are OK and hostname matches 'wall.alphacoders.com'.
	"""
	uri = urlparse(url)
	if uri.hostname != "wall.alphacoders.com":
		return False

	if "big.php" not in uri.path:
		return True

	tags = get_tags_alphacoders(html)

	log.debug("image tags: {}".format(tags), extra=log_extra)
	return not areTagsBlacklisted(tags)

def get_tags_alphacoders(html):
	soup = BeautifulSoup(html, 'html.parser')
	list_tags = soup.find(id="list_tags")
	tags = []
	for item in list_tags.find_all("a"):
		tag = item.contents[0]
		tags.append(tag.lower())
	return tags

def get_tags_wallpaperscraft(html):
	soup = BeautifulSoup(html, 'html.parser')
	list_tags = soup.find_all(class_="wb_tags")
	tags = []
	for item in list_tags.find_all("a"):
		tag = item.contents[0]
		tags.append(tag.lower())
	return tags

def get_highest_resolution_wallpaperscraft(html):
	soup = BeautifulSoup(html, 'html.parser')
	wb_more = soup.find_all("wb_more")
	for link in wb_more.find_all("a"):
		return link.get("href")

def parse_json(url, text):
	log_extra = {"pid":os.getpid(), "worker":"CRAWL"}
	uri = urlparse(url)
	root = json.loads(text)

	if "artstation.com" in uri.hostname:
		artstation_exclude_artists = ["tomgawronski"]
		for item in root["data"]:
			if not (item["views_count"] > 80 and item["likes_count"] > 1):
				log.warning("skipping artstation because not enough views or likes. {} views, {} likes, {}".format(item["views_count"], item["likes_count"], item["permalink"]), extra=log_extra)
				continue
			if item["user"]["username"] in artstation_exclude_artists:
				log.warning("skipping because excluded artstation artist {}".format(url), extra=log_extra)
				continue
			if item["hide_as_adult"]:
				log.warning("possible adult content {}".format(url), extra=log_extra)
			if not item["user"]["pro_member"]:
				log.warning("might lack quality {}".format(url), extra=log_extra)
			link = item["permalink"]
			if link:
				# log.debug("adding {}".format(link), extra=log_extra)
				add_crawl(link)
	else:
		log.error("can't parse json {}".format(url), extra=log_extra)


def download_image(url):
	assert isinstance(url, str) and len(url) > 0
	log_extra = {"pid":os.getpid(), "worker":"DOWNLOAD"}

	uri = urlparse(url)
	# if "deviantart" in uri.hostname:
	# 	log.debug("deviantart url cleaning", extra=log_extra)
	# 	regex_fit_in = re.compile("\/fit-in\/[0-9]+x[0-9]+")
	# 	uri._replace(path=regex_fit_in.sub("", url.path)) # remove matches

	log.info(uri.geturl(), extra=log_extra)
	file_name = uri.path.split("/")[-1]
	folder_path = download_location / uri.hostname
	full_path = folder_path.joinpath(file_name)
	if full_path.exists():
		log.warn("already exists {}".format(full_path), extra=log_extra)
		return True

	if not folder_path.exists():
		log.debug("creating {}".format(folder_path), extra=log_extra)
		folder_path.mkdir(parents=True)

	if "token" in uri.query:
		response = requests.get(url, timeout=8)
	else:
		response = requests.get(url.split("?")[0], timeout=8)

	if response.status_code == 200:
		# file_name = "img{}.{}".format(str(uuid.uuid4()), uri.path.split(".")[-1])
		log.info("saving to {}".format(full_path), extra=log_extra)
		with open(str(full_path), "wb") as f:
			for chunk in response.iter_content(chunk_size=256):
				f.write(chunk)

		check_image(full_path)
	else:
		log.error("failed ({}) {}".format(response.status_code, response.url), extra=log_extra)

def check_image(full_path):
	with Image.open(full_path) as im:
		log.debug("image size: {}".format(im.size), extra=log_extra)
		width, height = im.size
		# im.size is a tuple (width, height)
		# NOTE: tuple comparison doesn't behave the way we want: https://stackoverflow.com/questions/5292303/how-does-tuple-comparison-work-in-python
		if width < minimum_resolution[0] or height < minimum_resolution[1]:
			log.info("too small ({}), deleting {}".format(im.size, full_path), extra=log_extra)
			os.remove(str(full_path))
		elif height > width:
			log.warn("height > width ({}), deleting {}".format(im.size, full_path))
			os.remove(str(full_path))
		elif abs(width - height) < 250:
			log.warn("almost a square ({}), deleting {}".format(im.size, full_path))
			os.remove(str(full_path))


def handle_zip(full_path):
	# TODO
	pass

def worker_crawl(queue):
	if not queue.empty():
		crawl_page(queue.get())

def worker_download(queue):
	if not queue.empty():
		download_image(queue.get())

if __name__ == "__main__":
	log_extra = {"pid":os.getpid(), "worker":"MAIN"}
	# print("Working directory:", os.curdir)
	# print("CPU count", cpu_count())
	log.info("Working directory: {}".format(os.curdir), extra=log_extra)
	log.info("CPU count: {}".format(cpu_count()), extra=log_extra)

	m = Manager()

	# a queue of web page URLs to crawl for images
	crawl_queue = m.Queue(max_crawl_queue)
	# a queue of image URLs to download
	download_queue = m.Queue(max_download_queue)

	queued_urls = m.list()

	# TEST CASES
	# add_crawl("https://www.artstation.com/artwork/3b3VD")
	# add_crawl("https://www.artstation.com/artwork/Lvo4R")
	# add_crawl("https://www.artstation.com/artwork/kQVNz")
	# add_crawl("https://www.artstation.com/artwork/QY1dE")
	# add_crawl("https://wall.alphacoders.com/big.php?i=872081")
	# add_crawl("https://wall.alphacoders.com/big.php?i=72092")
	# add_crawl("https://wall.alphacoders.com/featured.php")

	add_crawl("https://www.artstation.com/search/projects.json?direction=desc&order=published_at&page=1&q=wallpaper&show_pro_first=true")
	# add_crawl("https://www.artstation.com/search/projects.json?direction=desc&order=likes_count&page=1&q=scifi&show_pro_first=true")
	add_crawl("https://www.artstation.com/search/projects.json?direction=desc&order=likes_count&page=1&q=sports%20car&show_pro_first=true")
	# add_crawl("https://www.artstation.com/search/projects.json?direction=desc&order=likes_count&page=1&q=space&show_pro_first=true")
	# add_crawl("https://www.artstation.com/search/projects.json?direction=desc&order=likes_count&page=1&q=scenery&show_pro_first=true")
	add_crawl("https://www.pexels.com/discover/")
	add_crawl("https://www.pexels.com/popular-photos/")
	add_crawl("https://www.pexels.com/search/HD%20wallpaper/")
	add_crawl("https://www.deviantart.com/art/The-Scent-of-the-Night-181412881")
	add_crawl("https://www.deviantart.com/customization/wallpaper/scenery/popular-all-time/")
	add_crawl("https://wallpaperscraft.com/")
	add_crawl("https://planwallpaper.com/")
	add_crawl("https://wallpaperbrowse.com/")
	add_crawl("https://wall.alphacoders.com/")
	add_crawl("https://wallpaperswide.com/")
	add_crawl("https://picswalls.com/")
	add_crawl("http://www.guoguiyan.com/")
	add_crawl("http://www.hd-wallpapers.me/")
	add_crawl("http://www.wallpapersafari.com/")
	add_crawl("http://www.hcxypz.com/group/hd-wallpapers/")
	add_crawl("http://www.allcoolwallpapers.com/")
	add_crawl("http://www.wallpapers-library.com/")

	# TEST CASES FOR EXCLUSION
	# add_crawl("https://www.deviantart.com/art/rei-ayanami-61656137")

	if not download_location.exists():
		log.debug("creating {}".format(download_location), extra=log_extra)
		download_location.mkdir(parents=True)

	# int(cpu_count() / 2)
	crawl_pool = Pool(2, worker_crawl, (crawl_queue,))
	download_pool = Pool(6, worker_download, (download_queue,))

	# crawl_pool.apply_async(worker_crawl, (crawl_queue,))
	# download_pool.apply_async(worker_download, (download_queue,))

	time.sleep(5)

	running_jobs = []

	while not crawl_queue.empty() or not download_queue.empty():
		for job in running_jobs:
			if job.ready():
				running_jobs.remove(job)

		log.info("crawl_queue: {:>5} | download_queue: {:>5} | Running Jobs: {:>5} | queued_urls: {:>5}".format(crawl_queue.qsize(), download_queue.qsize(), len(running_jobs), len(queued_urls)), extra=log_extra)
		if len(running_jobs) > cpu_count():
			log.debug("Too many jobs. Waiting...", extra=log_extra)
			time.sleep(0.7)
			continue

		if not crawl_queue.empty():
			result_crawl = crawl_pool.apply_async(worker_crawl, (crawl_queue,))
			running_jobs.append(result_crawl)
		if not download_queue.empty():
			result_download = download_pool.apply_async(worker_download, (download_queue,))
			running_jobs.append(result_download)
		# result_crawl.wait(timeout=30)
		# result_download.wait(timeout=30)
		# print("result_crawl:", result_crawl.ready(), "result_download:", result_download.ready())

		# running_jobs += [result_crawl, result_download]

	while len(running_jobs) > 0:
		for job in running_jobs:
			if job.ready():
				running_jobs.remove(job)
		time.sleep(0.2)

	log.info("saving queued_urls", extra=log_extra)
	with open("queued_urls.txt", "w") as f:
		f.write("\n".join(queued_urls))
	log.info("exiting loop", extra=log_extra)
