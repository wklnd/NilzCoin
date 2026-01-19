"""Simple Tkinter GUI for NilzCoin wallet management."""

from __future__ import annotations

import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from decimal import Decimal, InvalidOperation
from pathlib import Path
from argparse import Namespace
from typing import Dict, List, Optional

import requests

try:  # Support running as a module or as a script
    from .wallet import (
        DEFAULT_WALLET,
        WalletFile,
        WalletKey,
        craft_transaction,
        parse_amount,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    from wallet.wallet import (
        DEFAULT_WALLET,
        WalletFile,
        WalletKey,
        craft_transaction,
        parse_amount,
    )


DEFAULT_NODE_URL = "http://127.0.0.1:5000"
REQUEST_TIMEOUT = 10


class WalletGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NilzCoin Wallet")
        self.wallet: Optional[WalletFile] = None
        self.passphrase: Optional[str] = None
        self.keys: List[WalletKey] = []
        self.selected_key: Optional[WalletKey] = None
        self.address_summary: Optional[Dict] = None
        self._progress_visible = False
        self._ui_queue: "queue.Queue" = queue.Queue()
        self._build_ui()
        # Start UI queue polling on main thread
        self.root.after(50, self._drain_ui_queue)

    # UI construction --------------------------------------------------

    def _build_ui(self) -> None:
        self.wallet_path_var = tk.StringVar(value=str(DEFAULT_WALLET))
        self.node_url_var = tk.StringVar(value=DEFAULT_NODE_URL)
        self.passphrase_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Load a wallet to begin")
        self.balance_var = tk.StringVar(value="0")
        self.spendable_var = tk.StringVar(value="0")
        self.pending_var = tk.StringVar(value="0")
        self.selected_address_var = tk.StringVar(value="-")

        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Wallet file:").grid(row=0, column=0, sticky=tk.W)
        wallet_entry = ttk.Entry(top, textvariable=self.wallet_path_var, width=50)
        wallet_entry.grid(row=0, column=1, sticky=tk.EW, padx=5)
        ttk.Button(top, text="Browse", command=self._pick_wallet).grid(row=0, column=2, padx=5)

        ttk.Label(top, text="Node URL:").grid(row=1, column=0, sticky=tk.W)
        node_entry = ttk.Entry(top, textvariable=self.node_url_var, width=50)
        node_entry.grid(row=1, column=1, sticky=tk.EW, padx=5)

        ttk.Label(top, text="Passphrase:").grid(row=2, column=0, sticky=tk.W)
        pass_entry = ttk.Entry(top, textvariable=self.passphrase_var, show="*")
        pass_entry.grid(row=2, column=1, sticky=tk.EW, padx=5)
        ttk.Button(top, text="Load Wallet", command=self.load_wallet).grid(row=2, column=2, padx=5)

        top.columnconfigure(1, weight=1)

        body = ttk.Frame(self.root, padding=10)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(body, text="Addresses")
        left.pack(side=tk.LEFT, fill=tk.Y)
        self.address_listbox = tk.Listbox(left, height=15)
        self.address_listbox.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        self.address_listbox.bind("<<ListboxSelect>>", lambda _: self.on_address_select())
        scrollbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.address_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.address_listbox.config(yscrollcommand=scrollbar.set)

        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        info = ttk.LabelFrame(right, text="Address Details")
        info.pack(fill=tk.X)
        ttk.Label(info, text="Address:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(info, textvariable=self.selected_address_var, font=("Courier", 10)).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(info, text="Copy", command=self.copy_selected_address).grid(row=0, column=2, padx=5)
        ttk.Button(info, text="Refresh", command=self.refresh_selected_address).grid(row=0, column=3, padx=5)

        ttk.Label(info, text="Confirmed balance:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(info, textvariable=self.balance_var).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(info, text="Spendable:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(info, textvariable=self.spendable_var).grid(row=2, column=1, sticky=tk.W)
        ttk.Label(info, text="Pending outgoing:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Label(info, textvariable=self.pending_var).grid(row=3, column=1, sticky=tk.W)

        info.columnconfigure(1, weight=1)

        tx_frame = ttk.LabelFrame(right, text="Transactions")
        tx_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        columns = ("direction", "amount", "status", "timestamp", "txid")
        self.tx_tree = ttk.Treeview(tx_frame, columns=columns, show="headings", height=10)
        self.tx_tree.heading("direction", text="Dir")
        self.tx_tree.heading("amount", text="Amount")
        self.tx_tree.heading("status", text="Status")
        self.tx_tree.heading("timestamp", text="Timestamp")
        self.tx_tree.heading("txid", text="TxID")
        self.tx_tree.column("direction", width=60, anchor=tk.CENTER)
        self.tx_tree.column("amount", width=100)
        self.tx_tree.column("status", width=90)
        self.tx_tree.column("timestamp", width=160)
        self.tx_tree.column("txid", width=300)
        self.tx_tree.pack(fill=tk.BOTH, expand=True)

        send_frame = ttk.LabelFrame(right, text="Send Nilz")
        send_frame.pack(fill=tk.X)
        ttk.Label(send_frame, text="To address:").grid(row=0, column=0, sticky=tk.W)
        self.send_to_var = tk.StringVar()
        ttk.Entry(send_frame, textvariable=self.send_to_var, width=50).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        ttk.Label(send_frame, text="Amount:").grid(row=1, column=0, sticky=tk.W)
        self.send_amount_var = tk.StringVar()
        ttk.Entry(send_frame, textvariable=self.send_amount_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Button(send_frame, text="Send", command=self.send_transaction).grid(row=0, column=2, rowspan=2, padx=5)
        send_frame.columnconfigure(1, weight=1)

        status_frame = ttk.Frame(self.root, padding=(10, 0, 10, 5))
        status_frame.pack(fill=tk.X)
        self.status_bar = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=140)
        self.progress.pack(side=tk.RIGHT, padx=(10, 0))
        self.progress.stop()
        self.progress.pack_forget()

    # Wallet actions ---------------------------------------------------

    def _pick_wallet(self) -> None:
        initial = self.wallet_path_var.get() or str(DEFAULT_WALLET)
        path = filedialog.askopenfilename(initialdir=str(Path(initial).parent), title="Select wallet file")
        if path:
            self.wallet_path_var.set(path)

    def load_wallet(self) -> None:
        wallet_path = Path(self.wallet_path_var.get()).expanduser().resolve()
        passphrase = self.passphrase_var.get()
        if not passphrase:
            messagebox.showwarning("Passphrase required", "Enter the wallet passphrase before loading.")
            return
        try:
            wallet = WalletFile.load(wallet_path)
            wallet.unlock_signing_key(None, passphrase)
        except Exception as exc:
            messagebox.showerror("Failed to load", f"Could not open wallet: {exc}")
            self.set_status("Failed to load wallet")
            return
        self.wallet = wallet
        self.passphrase = passphrase
        self.keys = wallet.list_keys()
        self.populate_addresses()
        self.set_status(f"Loaded wallet with {len(self.keys)} address(es)")
        if self.keys:
            self.address_listbox.selection_set(0)
            self.on_address_select()

    def populate_addresses(self) -> None:
        self.address_listbox.delete(0, tk.END)
        for key in self.keys:
            self.address_listbox.insert(tk.END, f"{key.label} - {key.address}")

    def on_address_select(self) -> None:
        selection = self.address_listbox.curselection()
        if not selection or selection[0] >= len(self.keys):
            return
        key = self.keys[selection[0]]
        self.selected_key = key
        self.selected_address_var.set(key.address)
        self.clear_summary()
        self.refresh_selected_address()

    def clear_summary(self) -> None:
        self.balance_var.set("0")
        self.spendable_var.set("0")
        self.pending_var.set("0")
        self.address_summary = None
        for row in self.tx_tree.get_children():
            self.tx_tree.delete(row)

    # Data fetching -----------------------------------------------------

    def refresh_selected_address(self) -> None:
        if not self.selected_key:
            return
        node_url = self.node_url_var.get().strip()
        address = self.selected_key.address
        self._start_fetch(f"Fetching data for {address} ...")
        threading.Thread(
            target=self._fetch_address_summary,
            args=(node_url, address),
            daemon=True,
        ).start()

    def _fetch_address_summary(self, node_url: str, address: str) -> None:
        endpoint = node_url.rstrip("/") + f"/address/{address}"
        try:
            response = requests.get(endpoint, params={"limit": 200}, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            # Send error to main thread for handling
            self._ui_queue.put(("error", str(exc)))
            return
        # Send data to main thread for application
        self._ui_queue.put(("summary", data))

    def _drain_ui_queue(self) -> None:
        try:
            while True:
                kind, payload = self._ui_queue.get_nowait()
                if kind == "error":
                    self._handle_fetch_error(payload)
                elif kind == "summary":
                    self._apply_summary(payload)
        except queue.Empty:
            pass
        # Continue polling
        self.root.after(50, self._drain_ui_queue)

    def _handle_fetch_error(self, message: str) -> None:
        messagebox.showerror("Node request failed", message)
        self._hide_progress()
        self.set_status("Could not fetch address data")

    def _apply_summary(self, summary: Dict) -> None:
        # If address changed since request was initiated, discard and hide progress
        if not self.selected_key or summary.get("address") != self.selected_key.address:
            self._hide_progress()
            return
        self.address_summary = summary
        # Switch progress bar from network (indeterminate) to UI processing (determinate)
        self.progress.stop()
        self.progress.config(mode="determinate")
        self._show_progress()
        # Compute balances locally from transactions to be robust against API mismatches
        try:
            address = summary.get("address", "")
            txs = summary.get("transactions", [])
            confirmed_in = Decimal("0")
            confirmed_out = Decimal("0")
            pending_out = Decimal("0")
            for tx in txs:
                amt = Decimal(str(tx.get("amount", "0")))
                status = tx.get("status")
                if status == "confirmed":
                    if tx.get("type") == "coinbase" and tx.get("to") == address:
                        confirmed_in += amt
                    else:
                        if tx.get("to") == address:
                            confirmed_in += amt
                        if tx.get("from") == address:
                            confirmed_out += amt
                elif status == "pending":
                    if tx.get("from") == address and tx.get("type") != "coinbase":
                        pending_out += amt
            confirmed_balance = (confirmed_in - confirmed_out).quantize(Decimal("0.00000001"))
            spendable = max(confirmed_balance - pending_out, Decimal("0"))
            # Prefer node-provided values if present and non-empty, else use computed
            bal_str = summary.get("balance") or str(confirmed_balance)
            spend_str = summary.get("spendable") or str(spendable)
            pending_str = summary.get("pending_outgoing") or str(pending_out)
        except Exception:
            # Fallback to API values on any parsing error
            bal_str = summary.get("balance", "0")
            spend_str = summary.get("spendable", "0")
            pending_str = summary.get("pending_outgoing", "0")

        self.balance_var.set(str(bal_str))
        self.spendable_var.set(str(spend_str))
        self.pending_var.set(str(pending_str))
        # Clear existing transactions
        for row in self.tx_tree.get_children():
            self.tx_tree.delete(row)
        transactions = summary.get("transactions", [])
        total = len(transactions)
        if total == 0:
            # No transactions, finalize immediately
            self.set_status(f"Updated {summary.get('address')} (0 transactions)")
            self._hide_progress()
            return

        self.progress.config(maximum=total, value=0)
        self.set_status(f"Processing transactions 0/{total} (0%)")

        def insert_next(index: int) -> None:
            # Abort if selection changed mid-process
            if not self.selected_key or summary.get("address") != self.selected_key.address:
                self._hide_progress()
                return
            if index >= total:
                self.set_status(f"Updated {summary.get('address')} ({total} transactions)")
                self._hide_progress()
                return
            tx = transactions[index]
            direction = self._tx_direction(tx)
            self.tx_tree.insert(
                "",
                tk.END,
                values=(direction, tx.get("amount"), tx.get("status"), tx.get("timestamp"), tx.get("hash")),
            )
            self.progress['value'] = index + 1
            pct = int(((index + 1) / total) * 100)
            self.set_status(f"Processing transactions {index + 1}/{total} ({pct}%)")
            # Schedule next insertion without blocking UI
            self.root.after(5, lambda: insert_next(index + 1))

        # Kick off incremental insertion
        self.root.after(10, lambda: insert_next(0))

    def _tx_direction(self, tx: Dict) -> str:
        if not self.selected_key:
            return "?"
        if tx.get("type") == "coinbase" and tx.get("to") == self.selected_key.address:
            return "COINBASE"
        if tx.get("from") == self.selected_key.address:
            return "OUT"
        if tx.get("to") == self.selected_key.address:
            return "IN"
        return "?"

    # Sending -----------------------------------------------------------

    def send_transaction(self) -> None:
        if not self.wallet or not self.passphrase or not self.selected_key:
            messagebox.showwarning("Wallet not ready", "Load a wallet and select an address first.")
            return
        to_address = self.send_to_var.get().strip()
        amount_raw = self.send_amount_var.get().strip()
        if not to_address or not amount_raw:
            messagebox.showwarning("Missing data", "Enter both destination address and amount.")
            return
        try:
            amount = parse_amount(amount_raw)
        except (InvalidOperation, ValueError) as exc:
            messagebox.showerror("Invalid amount", f"{exc}")
            return
        args = Namespace(from_label=self.selected_key.label, to=to_address, amount=amount)
        try:
            tx = craft_transaction(self.wallet, args, self.passphrase)
        except Exception as exc:
            messagebox.showerror("Signing failed", f"Could not sign transaction: {exc}")
            return
        node_url = self.node_url_var.get().strip().rstrip("/")
        endpoint = node_url + "/tx"
        try:
            response = requests.post(endpoint, json=tx, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            messagebox.showerror("Broadcast failed", f"Node rejected transaction: {exc}")
            self.set_status("Broadcast failed")
            return
        messagebox.showinfo("Transaction submitted", f"Tx {tx['hash']} accepted by node")
        self.set_status(f"Broadcasted transaction {tx['hash']}")
        self.send_to_var.set("")
        self.send_amount_var.set("")
        self.refresh_selected_address()

    # Utilities ---------------------------------------------------------

    def copy_selected_address(self) -> None:
        if not self.selected_key:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.selected_key.address)
        self.set_status("Address copied to clipboard")

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _start_fetch(self, message: str) -> None:
        self.set_status(message)
        self._show_progress()

    def _show_progress(self) -> None:
        if not self._progress_visible:
            self.progress.pack(side=tk.RIGHT, padx=(10, 0))
            self._progress_visible = True
        self.progress.start(10)

    def _hide_progress(self) -> None:
        if self._progress_visible:
            self.progress.stop()
            self.progress.pack_forget()
            self._progress_visible = False


def main() -> None:
    root = tk.Tk()
    app = WalletGUI(root)
    root.geometry("1000x600")
    root.mainloop()


if __name__ == "__main__":
    main()
