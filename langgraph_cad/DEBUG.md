# راهنمای دیباگ — اصلاح معماری اولیهٔ LangGraph

این فایل دربارهٔ همان پروژهٔ اولیه است. شش node اصلی و نام‌هایشان حفظ شده‌اند. گزارش تغییرات دقیق و محدودیت‌ها در README و diffهای `verification/` آمده است.

## نخست چه چیزی را بررسی کنیم؟

| مشکل | محل بررسی |
|---|---|
| اطلاعات طراحی به کد نرسیده | payload در `code_generation`: request، requirements، cad_spec، geometry_plan و knowledge باید حاضر باشند |
| فرض عددی پنهان | requirements_original، پارامترهای assumed و script.assumptions |
| دانش ناکافی | selected_files، file_knowledge، knowledge.snippets و status=knowledge_insufficient |
| کد نمونه حذف شده | متن کامل note و code؛ محدودیت ۱۲ بخش و برش ۱۸۰۰ کاراکتر حذف شده‌اند |
| ایجنت به API بیشتری نیاز دارد | tool calls مربوط به search_knowledge/read_knowledge؛ منبع باید در catalog و داخل py_wiki باشد |
| فقط پروفایل تولید شده | run_result: document، اشیای نهایی، solid_count و volume |
| هندسه معتبر ولی اندازه اشتباه | validation.checks و geometry_plan.checks؛ مقدار CAD_PARAMETERS به‌تنهایی اندازه‌گیری نیست |
| خطای واقعی FreeCAD | attempt-*/run_result.json، traceback و stdout/stderr |
| شکست بازخوانی | document_reopened و step_roundtrip |
| تغییر تست برای عبور از شکست | revise_plan باید همان checks قبلی را حفظ کند؛ repair نمی‌تواند شکست را accept کند |
| اتصال شرکت پروکسی | OPENAI_BASE_URL، CAD_MODEL و پشتیبانی از tool/function calling |
| پروکسی embedding ندارد | سرویس جدا برای embedding، یا انتخاب صریح CAD_RETRIEVAL_MODE=lexical |

گراف پس از normalize_request ابعاد را بررسی می‌کند. مقدار معلوم با واحد نامعلوم حفظ می‌شود و سؤال واحد مطرح می‌شود. با --allow-assumptions، فقط گمشده‌ها به‌صورت فرض ثبت‌شده حل می‌شوند. Default اصلی، توقف برای روشن‌کردن اطلاعات است.

بعد از code_generation، run_freecad و validate_model اجرا می‌شوند. repair_model در موفقیت بدون فراخوانی مدل accept می‌کند؛ در شکست کد/پلن را برای اصلاح انتخاب می‌کند. حداکثر سه اصلاح پس از تلاش اولیه مجاز است. خطای محیط و timeout به‌عنوان مسئلهٔ طراحی به مدل واگذار نمی‌شوند.

FreeCAD در این محیط، مسیر Mod را هنگام embedding به prefix ناموجود اشاره می‌داد. worker اکنون از getResourceDir و CAD_FREECAD_MOD_DIR استفاده می‌کند؛ به همین دلیل PartDesign و Sketcher نیز در اجرای واقعی آزمایش شدند.

هیچ بدنهٔ دانش واقعی یا اعتبارنامهٔ پروکسی از شما در اختیار نبود. گراف، ابزارها و اجرای CAD آزموده شدند؛ ارزیابی end-to-end با مدل واقعی و دانش خودتان هنوز باید انجام شود. رکورد منبع و prompt نمی‌توانند تضمین کنند دانش پیش‌آموختهٔ مدل پاک شده است؛ این نسخه منبع را مرجع API قرار می‌دهد، منبع خوانده‌شده را ثبت می‌کند و نتیجه را واقعاً اجرا می‌کند.
