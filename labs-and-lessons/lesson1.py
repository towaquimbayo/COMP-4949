def ex1():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By

    def getBrowser():
        options = Options()

        # this parameter tells Chrome that
        # it should be run without UI (Headless)
        # Uncommment this line if you want to hide the browser.
        # options.add_argument('--headless=new')

        try:
            # initializing webdriver for Chrome with our options
            browser = webdriver.Chrome(options=options)
            print("Success.")
        except:
            print("It failed.")
        return browser

    browser = getBrowser()

    import time
    from bs4 import BeautifulSoup
    import re

    URL = "https://www.bcit.ca/study/programs/5512cert#courses"
    browser.get(URL)

    # Give the browser time to load all content.
    time.sleep(3)

    data = browser.find_elements(By.CSS_SELECTOR, ".clicktoshow")

    def getText(content):
        innerHtml = content.get_attribute("innerHTML")

        # Beautiful soup allows us to remove HTML tags from our content.
        soup = BeautifulSoup(innerHtml, features="lxml")
        rawString = soup.get_text()

        # Remove hidden carriage returns and tabs.
        textOnly = re.sub(r"[\n\t]*", "", rawString)
        # Replace two or more consecutive empty spaces with '*'
        textOnly = re.sub("[ ]{2,}", " ", textOnly)

        return textOnly

    for i in range(0, len(data)):
        text = getText(data[i])
        # date = getText(dates[i])
        print(str(i) + " " + text)
        print("***")  # Go to new line.


def ex2():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By

    def getBrowser():
        options = Options()

        # this parameter tells Chrome that
        # it should be run without UI (Headless)
        # Uncommment this line if you want to hide the browser.
        # options.add_argument('--headless=new')

        try:
            # initializing webdriver for Chrome with our options
            browser = webdriver.Chrome(options=options)
            print("Success.")
        except:
            print("It failed.")
        return browser

    browser = getBrowser()

    import time
    from bs4 import BeautifulSoup
    import re

    URL = "https://www.bcit.ca/study/programs/5512cert#courses"
    browser.get(URL)

    # Give the browser time to load all content.
    time.sleep(3)

    data = browser.find_elements(By.CSS_SELECTOR, ".clicktoshow , .course_number a")

    def getText(content):
        innerHtml = content.get_attribute("innerHTML")

        # Beautiful soup allows us to remove HTML tags from our content.
        soup = BeautifulSoup(innerHtml, features="lxml")
        rawString = soup.get_text()

        # Remove hidden carriage returns and tabs.
        textOnly = re.sub(r"[\n\t]*", "", rawString)
        # Replace two or more consecutive empty spaces with '*'
        textOnly = re.sub("[ ]{2,}", " ", textOnly)

        return textOnly

    for i in range(0, len(data), 2):
        course = getText(data[i])
        text = getText(data[i + 1])
        # date = getText(dates[i])
        print(course, text)
        print("Towa ***")  # Go to new line.


def ex3():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By

    def getBrowser():
        options = Options()

        # this parameter tells Chrome that
        # it should be run without UI (Headless)
        # Uncommment this line if you want to hide the browser.
        # options.add_argument('--headless=new')

        try:
            # initializing webdriver for Chrome with our options
            browser = webdriver.Chrome(options=options)
            print("Success.")
        except:
            print("It failed.")
        return browser

    browser = getBrowser()

    import time
    from bs4 import BeautifulSoup
    import re

    URL = "https://vpl.bibliocommons.com/events/search/index"
    browser.get(URL)

    # Give the browser time to load all content.
    time.sleep(3)

    data = browser.find_elements(By.CSS_SELECTOR, ".cp-events-search-item")

    def getText(content):
        innerHtml = content.get_attribute("innerHTML")

        # Beautiful soup allows us to remove HTML tags from our content.
        soup = BeautifulSoup(innerHtml, features="lxml")
        rawString = soup.get_text()

        # Remove hidden carriage returns and tabs.
        textOnly = re.sub(r"[\n\t]*", "", rawString)
        # Replace two or more consecutive empty spaces with '*'
        textOnly = re.sub("[ ]{2,}", " ", textOnly)

        return textOnly

    def getEndTime(content):
        amIdx = content.find("am")  # Get index of 1st 'am' occurence in string.
        pmIdx = content.find("pm")

        if amIdx >= 0 and (amIdx < pmIdx or pmIdx == -1):
            endTime = content[0:amIdx] + "am"  # add 'am' to substring
            return endTime
        startTime = content[0:pmIdx] + "pm"
        return startTime

    def getEventTitle(dayNumOfMonth, text):
        daysOfWeek = [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
        ]
        dayIndexes = []
        for day in daysOfWeek:
            dayIdx = text.find(day)
            if dayIdx >= 0:
                dayIndexes.append(dayIdx)
        dayIndexes.sort()
        startIndex = text.find(dayNumOfMonth) + len(dayNumOfMonth)
        title = text[startIndex : dayIndexes[0]]
        return title

    def getBranchLocation(text):
        # Find 'BranchEvent' in text and first occurence of ':' after it. The first index after ':' is the start of the branch location and the first occurrence of 'Branch' or 'Central Library' after that is the end of the branch location.
        branchIdx = text.find("BranchEvent")
        colonIdx = text.find(":", branchIdx)
        startIdx = colonIdx + 1
        endIdx = text.find("Branch", startIdx)
        if endIdx == -1:
            startIdx = text.find("Central Library", colonIdx + 1)
            endIdx = startIdx + len("Central Library")
            if startIdx == -1:
                startIdx = text.find("Online event", colonIdx + 1)
                endIdx = startIdx + len("Online event")
        else:
            endIdx += len("Branch")
        branchLocation = text[startIdx:endIdx]
        return branchLocation

    for i in range(0, len(data)):
        text = getText(data[i])
        textArray = text.split(",")

        print(str(i) + " " + text)
        DATE_IDX = 1
        YEAR_IDX = 2
        INFO_IDX = 3
        date = textArray[DATE_IDX].split("on ")[0]
        dayOfMonth = date.strip().split(" ")[1]
        year = textArray[YEAR_IDX].strip()  # strip() removes extra characters.
        startTime = textArray[INFO_IDX].split("–")[0]
        endTime = getEndTime(textArray[INFO_IDX].split("–")[1])
        title = getEventTitle(dayOfMonth, text)
        branchLocation = getBranchLocation(textArray[INFO_IDX])

        print("\n" + title)
        print(
            "Date: "
            + date
            + "   Year: "
            + year
            + " Start time: "
            + startTime
            + "  End time: "
            + endTime
        )
        print("Branch: " + branchLocation)
        print("***")  # Go to new line.


def ex5():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By

    def getBrowser():
        options = Options()

        # this parameter tells Chrome that
        # it should be run without UI (Headless)
        # Uncommment this line if you want to hide the browser.
        # options.add_argument('--headless=new')

        try:
            # initializing webdriver for Chrome with our options
            browser = webdriver.Chrome(options=options)
            print("Success.")
        except:
            print("It failed.")
        return browser

    browser = getBrowser()

    import time
    import re
    from bs4 import BeautifulSoup

    URL = "https://bpl.bc.ca/events/"
    browser.get(URL)

    # Give the browser time to load all content.
    time.sleep(1)

    SEARCH_TERM = "Analytics"
    search = browser.find_element(By.CSS_SELECTOR, "input")
    search.send_keys(SEARCH_TERM)

    # Find the search button - this is only enabled when a search query is entered
    button = browser.find_element(By.CSS_SELECTOR, "button")
    button.click()  # Click the button.
    time.sleep(3)

    def getContent(content):
        textContent = content.get_attribute("innerHTML")

        # Beautiful soup removes HTML tags from our content if it exists.
        soup = BeautifulSoup(textContent, features="lxml")
        rawString = soup.get_text().strip()

        # Remove hidden characters for tabs and new lines.
        rawString = re.sub(r"[\n\t]*", "", rawString)

        # Replace two or more consecutive empty spaces with '*'
        rawString = re.sub("[ ]{2,}", "*", rawString)
        return rawString

    # content = browser.find_elements_by_css_selector(".cp-search-result-item-content")
    pageNum = 1

    for i in range(0, 9):
        titles = browser.find_elements(By.CSS_SELECTOR, ".title-content")
        formats = browser.find_elements(
            By.CSS_SELECTOR, ".manifestation-item-format-info-wrap"
        )
        availability = browser.find_elements(
            By.CSS_SELECTOR, ".manifestation-item-availability-block-wrap"
        )

        NUM_ITEMS = len(titles)

        # This technique works only if counts of all scraped items match.
        if (
            len(titles) != NUM_ITEMS
            or len(formats) != NUM_ITEMS
            or len(availability) != NUM_ITEMS
        ):
            print("**WARNING: Items scraped are misaligned because their counts differ")

        for i in range(0, NUM_ITEMS):
            title = getContent(titles[i])
            mediaFormat = getContent(formats[i])
            available = getContent(availability[i])
            print("Title: " + title)
            print("Media: " + mediaFormat)
            print("Availability: " + available)
            print("********")

        # Go to a new page.
        pageNum += 1

        URL_NEXT = (
            "https://burnaby.bibliocommons.com/v2/search?query="
            + SEARCH_TERM
            + "&searchType=smart&pagination_page="
        )

        URL_NEXT = URL_NEXT + str(pageNum)
        browser.get(URL_NEXT)
        print("Count: ", str(i))
        time.sleep(3)

    browser.quit()
    print("done loop")


def ex6():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By

    def getBrowser():
        options = Options()

        # this parameter tells Chrome that
        # it should be run without UI (Headless)
        # Uncommment this line if you want to hide the browser.
        # options.add_argument('--headless=new')

        try:
            # initializing webdriver for Chrome with our options
            browser = webdriver.Chrome(options=options)
            print("Success.")
        except:
            print("It failed.")
        return browser

    browser = getBrowser()

    import time
    import re
    from bs4 import BeautifulSoup

    URL = "https://www.ebay.ca/"
    browser.get(URL)

    # Give the browser time to load all content.
    time.sleep(1)

    SEARCH_TERM = "ps5"
    search = browser.find_element(By.CSS_SELECTOR, "#gh-ac")
    search.send_keys(SEARCH_TERM)

    # Find the search button - this is only enabled when a search query is entered
    button = browser.find_element(By.CSS_SELECTOR, "button")
    button.click()  # Click the button.
    time.sleep(3)

    def getContent(content):
        textContent = content.get_attribute("innerHTML")

        # Beautiful soup removes HTML tags from our content if it exists.
        soup = BeautifulSoup(textContent, features="lxml")
        rawString = soup.get_text().strip()

        # Remove hidden characters for tabs and new lines.
        rawString = re.sub(r"[\n\t]*", "", rawString)

        # Replace two or more consecutive empty spaces with '*'
        rawString = re.sub("[ ]{2,}", "*", rawString)
        return rawString

    # content = browser.find_elements_by_css_selector(".cp-search-result-item-content")
    pageNum = 1

    for i in range(0, 2):
        titles = browser.find_elements(By.CSS_SELECTOR, ".s-item__title")
        # price = browser.find_elements(By.CSS_SELECTOR, ".lc-price")

        NUM_ITEMS = len(titles)

        # This technique works only if counts of all scraped items match.
        if len(titles) != NUM_ITEMS:
            print("**WARNING: Items scraped are misaligned because their counts differ")

        for i in range(0, NUM_ITEMS):
            title = getContent(titles[i])
            # mediaFormat = getContent(price[i])
            print("Title: " + title)
            # print("Price: " + mediaFormat)
            print("********")

        # Go to a new page.
        pageNum += 1
        URL_NEXT = (
            "https://www.ebay.ca/sch/i.html?_from=R40&_nkw="
            + SEARCH_TERM
            + "&_sacat=0&_pgn="
        )

        URL_NEXT = URL_NEXT + str(pageNum)
        browser.get(URL_NEXT)
        print("Count: ", str(i))
        time.sleep(3)

    browser.quit()
    print("done loop")


def main():
    # ex1()
    # ex2()
    ex3()
    # ex5()
    # ex6()


if __name__ == "__main__":
    main()
