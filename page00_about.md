---
layout: page
title: About
description: Information about the project, website, and links to the paper and SI
img: about.png # Add image post (optional)
caption: "A Serious Man (20??)  "
permalink: index.html
sidebar: true
---

---


# {{site.data.about.title}}
{{site.data.about.authors}}

{% for entry in site.data.about %}

{% if entry[0] != 'title' %}
{% if entry[0] != 'authors' %}
{% if entry[0] != 'image' %}
## {{entry[0]}}
{{entry[1]}}

{% if entry[0] == 'image' %}
<img src="{{site.url}}/{{site.baseurl}}/assets/img/{{site.data.about.image}}" >
{% endif %}

{% endif %}
{% endif %}
{% endif %}



{% endfor %}
