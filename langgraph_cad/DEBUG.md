# بررسی جریان تولید کد و مصرف توکن

این نسخه تا تولید Python پیش می‌رود. nodeهای اجرا، اعتبارسنجی و repair حذف شده‌اند.

| موضوع | محل بررسی |
|---|---|
| ایندکس کامل دانش | `python main.py --index-only` و `index_stats` |
| هزینهٔ آماده‌سازی تکراری | `changed_files` در اجرای دوم باید صفر باشد؛ `embedded_sections` نیز صفر |
| بخش‌های بسیار بزرگ | `oversized_embedding_sections` و `retrieval_stats.blocked_sections`؛ هیچ تقسیم بر اندازه انجام نمی‌شود |
| دانش انتخاب‌شده | `selection_candidates`، `file_knowledge` و شناسهٔ بخش/هش منبع |
| دانش تکراری | متن فقط در `file_knowledge[].text` است؛ فیلد تکراری code/note وجود ندارد |
| مقدار واقعی توکن | `generation_stats.calls[].input_tokens/output_tokens` از گزارش سرویس |
| بودجهٔ پیش از فراخوانی | `estimated_input`، `counter` و `estimated_input_total` |
| شمارندهٔ محافظه‌کارانه | `utf8_bytes_upper_bound` یعنی tokenizer کش نشده؛ README روش آماده‌سازی را توضیح می‌دهد |
| شکست بودجه | `status=token_budget_exceeded` و `errors` |
| فرض‌ها | `requirements_original`، پارامترهای assumed، `script.assumptions` |
| کد خروجی | `generated_code.py`؛ وضعیت `code_ready` تأیید هندسه نیست |
| منابع غایب | `knowledge_insufficient`؛ دانش ساختگی یا قالب جایگزین استفاده نمی‌شود |

در graph.invoke، هر اجرای تازه از select_files ایندکس را برای فایل‌های تغییرکرده به‌روزرسانی می‌کند. اجرای CLI نیز خودکار این کار را می‌کند. کاتالوگ اولیه نگه‌داری می‌شود؛ خود متن همهٔ فایل‌های Markdown مبنای جست‌وجو است.

اجرای اول برای کل دانش embedding می‌زند؛ این هزینه در توکن‌های چت جمع نمی‌شود. در اجراهای بعد، embedding محتوا و query کش می‌شود. فراخوانی مدل چت همچنان متن دانش انتخاب‌شده را مصرف می‌کند؛ ذخیرهٔ بردار، متن ورودی مدل را رایگان نمی‌کند.

دیباگ هندسه و فیدبک اجرای FreeCAD فعلاً فعال نیستند. برای تشخیص صحت مدل باید کد خروجی را جداگانه در FreeCAD اجرا و بررسی کنید.
