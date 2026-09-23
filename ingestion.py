"""Локальный импорт таблицы переводов в проверяемый вход существующего ядра.

CSV читается как текст (UTF-8, в том числе BOM), без угадывания типов gid.
Номер ошибки CSV включает заголовок; для Parquet нумерация начинается с 1.
Сопоставление полей подтверждает пользователь. Исходные строки не дедуплицируются.
"""

import csv
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import re
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_ROWS = 100_000
MAX_NODES = 10_000
MAX_COLUMNS = 256
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_ERRORS = 100
PREVIEW_ROWS = 5
PREVIEW_VALUE_LIMIT = 300
REQUIRED_FIELDS = ("src", "dst", "sum_kzt", "date")
OPTIONAL_FIELDS = ("currency", "transaction_id")
ALIASES = {
    "src": ("src", "sender_id", "source_id", "payer_id", "sender", "payer", "source", "from",
            "отправитель", "плательщик", "счет отправителя", "от кого"),
    "dst": ("dst", "recipient_id", "receiver_id", "destination_id", "recipient", "receiver", "destination", "to",
            "получатель", "счет получателя", "кому"),
    "sum_kzt": ("sum_kzt", "amount_kzt", "amount", "sum", "сумма", "сумма перевода", "сумма тенге"),
    "date": ("date", "transaction_date", "payment_date", "timestamp", "дата", "дата перевода", "дата платежа"),
    "currency": ("currency", "currency_code", "валюта", "код валюты"),
    "transaction_id": ("transaction_id", "tx_id", "payment_id", "id", "номер операции", "идентификатор операции"),
}


class ImportValidationError(ValueError):
    """Ошибки импорта для API: не более 100 записей row/column/message."""

    def __init__(self, errors):
        self.errors = list(errors)[:MAX_ERRORS]
        first = self.errors[0] if self.errors else {"row": None, "column": "file", "message": "Некорректный импорт"}
        location = f"строка {first['row']}, " if first["row"] is not None else ""
        super().__init__(f"{location}{first['column']}: {first['message']}")


def _error(row, column, message):
    return {"row": row, "column": column, "message": message}


def _fail(column, message, row=None):
    raise ImportValidationError([_error(row, column, message)])


def _is_missing(value):
    return value is None or value is pd.NA or value is pd.NaT or (
        isinstance(value, float) and math.isnan(value)
    )


def _read_upload(path, original_name=None):
    path = Path(path)
    if original_name is not None and Path(original_name).suffix.lower() not in (".csv", ".parquet"):
        _fail("file", "Поддерживаются только CSV и Parquet")
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            _fail("file", "Файл превышает ограничение 32 МиБ")
        with path.open("rb") as handle:
            parquet = handle.read(4) == b"PAR1"
        if original_name is not None and (Path(original_name).suffix.lower() == ".parquet") != parquet:
            _fail("file", "Содержимое файла не соответствует расширению CSV/Parquet")
        if parquet:
            parquet_file = pq.ParquetFile(path)
            try:
                if parquet_file.metadata.num_columns > MAX_COLUMNS:
                    _fail("columns", f"Допускается не более {MAX_COLUMNS} колонок")
                if parquet_file.metadata.num_rows > MAX_ROWS:
                    _fail("file", f"В локальном режиме допускается не более {MAX_ROWS:,} строк".replace(",", " "))
                uncompressed = sum(parquet_file.metadata.row_group(index).total_byte_size
                                   for index in range(parquet_file.metadata.num_row_groups))
                if uncompressed > MAX_UNCOMPRESSED_BYTES:
                    _fail("file", "Распакованный Parquet превышает ограничение 256 МиБ")
                if any(pa.types.is_nested(field.type) for field in parquet_file.schema_arrow):
                    _fail("columns", "Вложенные поля Parquet (списки, структуры, map) не поддерживаются")
            finally:
                parquet_file.close()
            frame = pd.read_parquet(path)
            row_numbers = list(range(1, len(frame) + 1))
            file_format = "parquet"
        else:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                sample = handle.read(65536)
                handle.seek(0)
                try:
                    delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
                except csv.Error:
                    first_line = sample.splitlines()[0] if sample.splitlines() else ""
                    delimiter = max((",", ";", "\t"), key=first_line.count)
                reader = csv.reader(handle, delimiter=delimiter, strict=True)
                header = next(reader, None)
                if header is None:
                    _fail("file", "Файл пуст")
                columns = [str(value).strip() for value in header]
                _validate_columns(columns)
                rows, row_numbers, errors = [], [], []
                while True:
                    row_number = reader.line_num + 1
                    row = next(reader, None)
                    if row is None:
                        break
                    if not row:
                        continue
                    if len(rows) + len(errors) >= MAX_ROWS:
                        _fail("file", f"В локальном режиме допускается не более {MAX_ROWS:,} строк".replace(",", " "), row_number)
                    if len(row) != len(columns):
                        errors.append(_error(row_number, "file", "Число полей отличается от заголовка; проверьте разделитель и кавычки"))
                        if len(errors) >= MAX_ERRORS:
                            raise ImportValidationError(errors)
                        continue
                    rows.append(row)
                    row_numbers.append(row_number)
                if errors:
                    raise ImportValidationError(errors)
            frame = pd.DataFrame(rows, columns=columns)
            file_format = "csv"
    except ImportValidationError:
        raise
    except UnicodeError:
        _fail("file", "CSV должен быть в кодировке UTF-8 (BOM допускается)")
    except (OSError, csv.Error, ValueError, TypeError) as error:
        if isinstance(error, csv.Error):
            _fail("file", "Некорректные кавычки или слишком длинное поле CSV")
        _fail("file", "Не удалось прочитать таблицу CSV/Parquet; проверьте файл")
    frame.columns = [str(column).strip() for column in frame.columns]
    _validate_columns(list(frame.columns))
    if frame.empty:
        _fail("file", "Таблица не содержит переводов")
    return frame, row_numbers, file_format


def _validate_columns(columns):
    if len(columns) > MAX_COLUMNS:
        _fail("columns", f"Допускается не более {MAX_COLUMNS} колонок", 1)
    if not columns or any(not column for column in columns):
        _fail("columns", "Все колонки должны иметь непустые названия", 1)
    if len(set(columns)) != len(columns):
        _fail("columns", "Названия колонок должны быть уникальными", 1)


def _column_key(value):
    return re.sub(r"[^\w]", "", value.casefold().replace("ё", "е")).replace("_", "")


def inspect_upload(path, original_name):
    """Инвентаризация перед подтверждением mapping; никаких файлов не создаёт."""
    frame, _, file_format = _read_upload(path, original_name)
    lookup = {}
    for column in frame.columns:
        lookup.setdefault(_column_key(column), column)
    suggested = {}
    for field, aliases in ALIASES.items():
        for alias in aliases:
            if _column_key(alias) in lookup:
                suggested[field] = lookup[_column_key(alias)]
                break
    def preview_value(value):
        text = "" if _is_missing(value) else str(value)
        return text if len(text) <= PREVIEW_VALUE_LIMIT else text[:PREVIEW_VALUE_LIMIT - 1] + "…"

    preview = [
        {str(column): preview_value(value) for column, value in zip(frame.columns, row)}
        for row in frame.head(PREVIEW_ROWS).itertuples(index=False, name=None)
    ]
    return {"columns": list(frame.columns), "preview": preview, "row_count": len(frame),
            "suggested_mapping": suggested, "format": file_format}


def _identifier(value):
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError("Идентификатор должен быть текстом или целым числом; float запрещён из-за потери точности")
    text = str(value).strip()
    if not text:
        raise ValueError("Идентификатор не заполнен")
    if len(text) > 256 or any(ord(char) < 32 for char in text):
        raise ValueError("Идентификатор длиннее 256 символов или содержит управляющие символы")
    return text


def _amount(value):
    if isinstance(value, bool) or _is_missing(value):
        raise ValueError("Нужна положительная конечная сумма в KZT")
    text = str(value).strip()
    if not re.fullmatch(r"[+\-]?(?:\d+|\d{1,3}(?:[ \u00a0\u202f]\d{3})+)(?:[.,]\d+)?", text):
        raise ValueError("Сумма: цифры, точка или запятая для дробной части, пробелы между тысячами")
    try:
        amount = Decimal(re.sub(r"[ \u00a0\u202f]", "", text).replace(",", "."))
        if not amount.is_finite() or amount <= 0:
            raise ValueError("Сумма должна быть положительной и конечной")
        if amount != amount.quantize(Decimal("0.01")):
            raise ValueError("Сумма должна содержать не более двух знаков после запятой; округление не выполняется")
        numeric = float(amount)
        if not math.isfinite(numeric) or Decimal(str(numeric)) != amount:
            raise ValueError("Сумма слишком велика для сохранения точности; проверьте значение")
        return numeric
    except InvalidOperation:
        raise ValueError("Некорректная или слишком большая сумма") from None


def _date(value):
    if _is_missing(value) or isinstance(value, (bool, int, float)):
        raise ValueError("Нужна дата YYYY-MM-DD или ДД.ММ.ГГГГ; числовые даты не поддерживаются")
    if isinstance(value, (date, datetime, pd.Timestamp)):
        parsed = pd.Timestamp(value)
    else:
        text = str(value).strip()
        if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}(?: \d{2}:\d{2}(?::\d{2})?)?", text):
            fmt = "%d.%m.%Y" + (" %H:%M:%S" if text.count(":") == 2 else " %H:%M" if ":" in text else "")
            try:
                parsed = pd.Timestamp(datetime.strptime(text, fmt))
            except (ValueError, OverflowError):
                raise ValueError("Некорректная календарная дата или время") from None
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?(?:Z|[+\-]\d{2}:\d{2})?)?", text):
            try:
                parsed = pd.Timestamp(text)
            except (ValueError, OverflowError):
                raise ValueError("Некорректная календарная дата или время") from None
        else:
            raise ValueError("Дата должна быть YYYY-MM-DD или ДД.ММ.ГГГГ; допускается время ЧЧ:ММ:СС")
    if pd.isna(parsed):
        raise ValueError("Дата не заполнена")
    try:
        parsed = parsed.as_unit("ns")
    except (OverflowError, ValueError):
        raise ValueError("Дата выходит за поддерживаемый диапазон") from None
    return parsed.tz_convert("UTC") if parsed.tzinfo is not None else parsed


def _metadata(metadata):
    if not isinstance(metadata, dict):
        _fail("metadata", "Укажите параметры набора данных")
    name = str(metadata.get("name") or "").strip()
    source = str(metadata.get("source") or "").strip()
    if not name or len(name) > 200:
        _fail("name", "Название набора должно содержать от 1 до 200 символов")
    if len(source) > 500:
        _fail("source", "Описание источника не должно превышать 500 символов")
    coverage = metadata.get("coverage", "unknown")
    if coverage not in ("unknown", "complete"):
        _fail("coverage", "Полнота должна быть unknown или complete")
    currency = str(metadata.get("currency", "KZT")).strip().upper()
    if currency != "KZT":
        _fail("currency", "В этой версии поддерживается только KZT; конвертация валют не выполняется")
    seed_values = metadata.get("seed_gids", [])
    if not isinstance(seed_values, list):
        _fail("seed_gids", "Список известных клиентов должен быть массивом строк gid")
    try:
        seeds = sorted({_identifier(gid) for gid in seed_values})
    except ValueError as error:
        _fail("seed_gids", str(error))
    if len(seeds) > MAX_NODES:
        _fail("seed_gids", f"Допускается не более {MAX_NODES} узлов")
    result = {"name": name, "source": source, "coverage": coverage, "currency": currency, "seed_gids": seeds}
    for field in ("period_from", "period_to"):
        value = metadata.get(field)
        if value in (None, ""):
            result[field] = None
            continue
        try:
            if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError
            result[field] = date.fromisoformat(value).isoformat()
        except ValueError:
            _fail(field, "Граница периода должна быть датой YYYY-MM-DD")
    if result["period_from"] and result["period_to"] and result["period_from"] > result["period_to"]:
        _fail("period_to", "Конец периода предшествует началу")
    return result


def build_dataset(upload_path, mapping, metadata, data_dir):
    """Проверить импорт и записать 3 Parquet + metadata.json, вернуть summary.

    depth=0 — исключительно совместимая с ядром заглушка: depth_known=False.
    coverage=complete — заявление пользователя о полноте за указанный период.
    period_from/to ограничивают допустимые даты, строки не удаляются молча.
    """
    frame, row_numbers, file_format = _read_upload(upload_path)
    clean_metadata = _metadata(metadata)
    if not isinstance(mapping, dict):
        _fail("mapping", "Подтвердите соответствие колонок")
    mapped = {key: value for key, value in mapping.items() if value not in (None, "")}
    mapping_errors = []
    for field in REQUIRED_FIELDS:
        if field not in mapped:
            mapping_errors.append(_error(None, field, "Выберите исходную колонку"))
    for field, column in mapped.items():
        if field not in REQUIRED_FIELDS + OPTIONAL_FIELDS:
            mapping_errors.append(_error(None, str(field), "Неизвестное поле сопоставления"))
        elif not isinstance(column, str) or column not in frame.columns:
            mapping_errors.append(_error(None, field, "Выбранной колонки нет в файле"))
    if mapping_errors:
        raise ImportValidationError(mapping_errors)
    if len(set(mapped.values())) != len(mapped):
        _fail("mapping", "Для разных полей выберите разные исходные колонки")
    positions = {field: frame.columns.get_loc(column) for field, column in mapped.items()}
    errors, records, seen_ids = [], [], {}
    gids = set(clean_metadata["seed_gids"])
    timezone_kind = None
    for row_number, row in zip(row_numbers, frame.itertuples(index=False, name=None)):
        parsed = {}
        for field in REQUIRED_FIELDS + OPTIONAL_FIELDS:
            if field not in positions:
                continue
            value = row[positions[field]]
            try:
                if field in ("src", "dst", "transaction_id"):
                    parsed[field] = _identifier(value)
                elif field == "sum_kzt":
                    parsed[field] = _amount(value)
                elif field == "date":
                    parsed[field] = _date(value)
                    kind = parsed[field].tzinfo is not None
                    if timezone_kind is None:
                        timezone_kind = kind
                    elif kind != timezone_kind:
                        raise ValueError("Смешаны даты с часовым поясом и без него; приведите их к одному формату")
                    calendar_date = parsed[field].date().isoformat()
                    if clean_metadata["period_from"] and calendar_date < clean_metadata["period_from"]:
                        raise ValueError("Дата раньше заявленного начала периода")
                    if clean_metadata["period_to"] and calendar_date > clean_metadata["period_to"]:
                        raise ValueError("Дата позже заявленного конца периода")
                else:
                    if str(value).strip().upper() != "KZT":
                        raise ValueError("Поддерживается только KZT; другие валюты не конвертируются")
                    parsed[field] = "KZT"
            except (ValueError, TypeError, OverflowError) as error:
                errors.append(_error(row_number, mapped[field], str(error)))
                if len(errors) >= MAX_ERRORS:
                    raise ImportValidationError(errors)
        txid = parsed.get("transaction_id")
        if txid is not None:
            if txid in seen_ids:
                errors.append(_error(row_number, mapped["transaction_id"], f"Повтор transaction_id; первое появление в строке {seen_ids[txid]}"))
            else:
                seen_ids[txid] = row_number
        if "src" in parsed and "dst" in parsed:
            gids.update((parsed["src"], parsed["dst"]))
            if len(gids) > MAX_NODES:
                errors.append(_error(row_number, "gid", f"В локальном режиме допускается не более {MAX_NODES:,} уникальных узлов".replace(",", " ")))
                raise ImportValidationError(errors)
        records.append(parsed)
        if len(errors) >= MAX_ERRORS:
            raise ImportValidationError(errors)
    if errors:
        raise ImportValidationError(errors)
    tx = pd.DataFrame(records)
    tx["date"] = pd.to_datetime(tx.date)
    tx["currency"] = "KZT"
    edges = tx.groupby(["src", "dst"], sort=True, as_index=False).agg(
        sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size"))
    if not edges.sum_kzt.map(math.isfinite).all() or not math.isfinite(float(tx.sum_kzt.sum())):
        _fail("sum_kzt", "Суммарный оборот слишком велик для расчёта")
    edges["depth"] = 0
    seed_set = set(clean_metadata["seed_gids"])
    nodes = pd.DataFrame({"gid": sorted(gids)})
    nodes["depth"] = 0
    nodes["is_seed"] = nodes.gid.isin(seed_set)
    nodes["depth_known"] = False
    nodes["outgoing_coverage_known"] = clean_metadata["coverage"] == "complete"
    nodes["seed_incoming_incomplete"] = False
    warnings = ["Колено обхода неизвестно; техническое depth=0 не означает нулевой шаг"]
    if clean_metadata["coverage"] == "unknown":
        warnings.append("Полнота исходящих неизвестна: роли удержания средств не назначаются")
    if timezone_kind:
        warnings.append("Даты с часовым поясом приведены к UTC; границы периода проверены в UTC")
    summary = {
        "row_count": len(tx), "node_count": len(nodes), "edge_count": len(edges),
        "seed_count": int(nodes.is_seed.sum()), "total_kzt": float(tx.sum_kzt.sum()),
        "period_from": tx.date.min().date().isoformat(), "period_to": tx.date.max().date().isoformat(),
        "currency": "KZT", "coverage": clean_metadata["coverage"], "warnings": warnings,
    }
    document = {**clean_metadata, "schema_version": 1, "format": file_format,
                "mapping": mapped, "summary": summary}
    with Path(upload_path).open("rb") as handle:
        document["source_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    document["source_bytes"] = Path(upload_path).stat().st_size
    data_dir = Path(data_dir)
    filenames = ("nodes.parquet", "edges.parquet", "transactions.parquet", "metadata.json")
    if any((data_dir / filename).exists() for filename in filenames):
        _fail("data_dir", "Каталог уже содержит набор данных; выберите новый каталог, чтобы сохранить историю")
    data_dir.mkdir(parents=True, exist_ok=True)
    # Все файлы сначала успешно сериализуются, после чего становятся входом ядра.
    with tempfile.TemporaryDirectory(prefix=".ingestion-", dir=data_dir) as staging:
        staging = Path(staging)
        nodes.to_parquet(staging / "nodes.parquet", index=False)
        edges.to_parquet(staging / "edges.parquet", index=False)
        tx.to_parquet(staging / "transactions.parquet", index=False)
        (staging / "metadata.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        for filename in filenames:
            (staging / filename).replace(data_dir / filename)
    return summary
