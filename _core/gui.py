import sys
import tkinter as tk
from tkinter import filedialog, messagebox #, ttk
from uuid import uuid4
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mplfinance.original_flavor import candlestick_ohlc
from generator import *
from handle_csv_data import process_all_files, process_file
from share import *
import ttkbootstrap as ttk
from database import get_cassandra_session



class NewShare:
    def __init__(self, parent, session):
        self.parent = parent
        self.session = session
        self.entries = {}
        self.parameter_entries = []
        self.parameter_labels = []
        self.tickers = self.get_all_tickers()
        self.function_types = NoiseFunction.get_noise_function_types()
        self.last_saved_ticker = None

        # Frames for organizing UI components
        self.share_frame = ttk.Frame(self.parent)
        self.noise_frame = ttk.Frame(self.parent)

        # Initialize UI components
        self.create_share_widgets()
        self.create_noise_widgets()

        # Initially disable noise frame
        self.noise_frame.pack_forget()
        
        self.share_frame.pack(padx=5, pady=20)

    def create_share_widgets(self):
        fields = ['share_ticker:', 'share_name:', 'share_description:', 'share_value:', 'share_events:']
        for i, field in enumerate(fields):
            label = ttk.Label(self.share_frame, text=field)
            label.grid(row=i, column=0, sticky='w', padx=2, pady=2)
            entry = ttk.Entry(self.share_frame)
            entry.grid(row=i, column=1, padx=2, pady=2)
            self.entries[field] = entry
        submit_button = ttk.Button(self.share_frame, text="Add New Share", command=self.add_share)
        submit_button.grid(row=len(fields)+1, column=0, columnspan=2, padx=2, pady=2)

    def add_share(self):
        share_details = {field[:-1]: self.entries[field].get() for field in self.entries}
        if not share_details['share_ticker'] or not share_details['share_value']:
            messagebox.showwarning("Missing Field", "share_ticker and share_value fields cannot be empty or None.")
            return

        share_details['share_name'] = share_details['share_name'] or share_details['share_ticker']
        share_details['share_description'] = share_details['share_description'] or share_details['share_name']
        share_details['share_events'] = share_details['share_events'] or {}
        share_details['share_value'] = float(share_details['share_value'])

        query = self.session.prepare("INSERT INTO shares.share_details (share_ticker, share_name, share_description, share_value, share_events) VALUES (?, ?, ?, ?, ?);")
        self.session.execute(query, tuple(share_details.values()))
        messagebox.showinfo("Success", "New share has been added to the database")

        self.last_saved_ticker = share_details['share_ticker']
        self.noise_frame.pack(padx=5, pady=100)
        self.update_noise_widgets_for_ticker(self.last_saved_ticker)

        for entry in self.entries.values():
            entry.delete(0, 'end')

    def create_noise_widgets(self):
        self.ticker_var = tk.StringVar()
        self.function_type_var = tk.StringVar()

        ticker_menu = ttk.OptionMenu(self.noise_frame, self.ticker_var, '', *self.tickers)
        ticker_menu.grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(self.noise_frame, text="Ticker:").grid(row=0, column=0, padx=2, pady=2)

        function_type_menu = ttk.OptionMenu(self.noise_frame, self.function_type_var, '', *self.function_types, command=self.display_parameter_widgets)
        function_type_menu.grid(row=1, column=1, padx=2, pady=2)
        ttk.Label(self.noise_frame, text="Function Type:").grid(row=1, column=0, padx=2, pady=2)

        save_button = ttk.Button(self.noise_frame, text="Save Noise", command=self.save_noise)
        save_button.grid(row=58, column=0, columnspan=2, padx=2, pady=2)

    def update_noise_widgets_for_ticker(self, ticker):
        self.ticker_var.set(ticker)
        self.function_type_var.set(self.function_types[0] if self.function_types else '')

        ticker_menu = self.noise_frame.winfo_children()[1]
        function_type_menu = self.noise_frame.winfo_children()[3]
        save_button = self.noise_frame.winfo_children()[4]

        ticker_menu.config(state='normal')
        function_type_menu.config(state='normal')
        save_button.config(state='normal')

        self.display_parameter_widgets()

    def display_parameter_widgets(self, function_type=None, initial_params=None):
        for widget in self.parameter_entries + self.parameter_labels:
            widget.grid_forget()
        self.parameter_entries.clear()
        self.parameter_labels.clear()

        parameter_specs = NoiseFunction.get_parameter_spec(function_type or self.function_type_var.get())
        for i, param in enumerate(parameter_specs):
            label = ttk.Label(self.noise_frame, text=f"{param}:")
            label.grid(row=i+2, column=0, padx=1, pady=1)
            self.parameter_labels.append(label)

            entry = ttk.Entry(self.noise_frame)
            if initial_params and param in initial_params:
                entry.insert(0, initial_params[param])
            entry.grid(row=i+2, column=1, padx=1, pady=1)
            self.parameter_entries.append(entry)

    def get_all_tickers(self):
        rows = self.session.execute('SELECT share_ticker FROM shares.share_details;')
        return sorted(row.share_ticker for row in rows)
    
    def save_noise(self):
        ticker = self.ticker_var.get()
        function_type = self.function_type_var.get()
        noise_params = {NoiseFunction.get_parameter_spec(function_type)[i]: entry.get() for i, entry in enumerate(self.parameter_entries)}
        query = self.session.prepare("INSERT INTO shares.noise_functions (share_ticker, noise_function_type, noise_parameters) VALUES (?, ?, ?);")
        try:
            self.session.execute(query, (ticker, function_type, noise_params))
            messagebox.showinfo("Success", "Noise has been saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save noise. Error: {e}")

        for entry in self.parameter_entries:
            entry.delete(0, 'end')



class AddNoise:
    def __init__(self, parent, session):
        self.parent = parent
        self.session = session
        self.tickers = self.get_all_tickers()
        self.function_types = NoiseFunction.get_noise_function_types()
        self.parameter_entries = []
        self.parameter_labels = []
        self.create_widgets()

    def get_all_tickers(self):
        rows = self.session.execute('SELECT share_ticker FROM shares.share_details;')
        return sorted(row.share_ticker for row in rows)

    def _display_parameter_widgets(self, function_type=None, initial_params=None):
        # Clear existing widgets
        for widget in self.parameter_entries + self.parameter_labels:
            widget.grid_forget()
        self.parameter_entries.clear()
        self.parameter_labels.clear()

        # Fetch parameter specs and display new widgets
        parameter_specs = NoiseFunction.get_parameter_spec(function_type or self.function_type_var.get())
        for i, param in enumerate(parameter_specs):
            label = ttk.Label(self.parent, text=f"{param}:")
            label.grid(row=i+2, column=0, padx=1, pady=1)
            self.parameter_labels.append(label)

            entry = ttk.Entry(self.parent)
            if initial_params and param in initial_params:
                entry.insert(0, initial_params[param])
            entry.grid(row=i+2, column=1, padx=1, pady=1)
            self.parameter_entries.append(entry)

    def _fetch_noise_function(self, selected_ticker=None):
        ticker = selected_ticker or self.ticker_var.get()
        query = 'SELECT * FROM shares.noise_functions WHERE share_ticker = %s'
        rows = self.session.execute(query, (ticker,))
        row = rows.one()  # Gets the first row from the result set
        if row:
            self.function_type_var.set(row.noise_function_type)
            # Assuming noise_parameters is a dictionary
            self._display_parameter_widgets(row.noise_function_type, row.noise_parameters)
        else:
            # Set to default if no function configuration exists
            self.function_type_var.set(self.function_types[0])
            self._display_parameter_widgets(self.function_types[0])


    def create_widgets(self):
        self.ticker_var = tk.StringVar(self.parent)
        self.function_type_var = tk.StringVar(self.parent)
        
        # Set the initial ticker
        self.ticker_var.set(self.tickers[0] if self.tickers else '')
        ticker_menu = ttk.OptionMenu(self.parent, self.ticker_var, self.ticker_var.get(), *self.tickers, command=self._fetch_noise_function)
        ticker_menu.grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(self.parent, text="Ticker:").grid(row=0, column=0, padx=2, pady=2)

        # Set the initial function type and create its menu
        initial_type = self.function_types[0] if self.function_types else ''
        self.function_type_var.set(initial_type)
        function_type_menu = ttk.OptionMenu(self.parent, self.function_type_var, initial_type, *self.function_types, command=self._display_parameter_widgets)
        function_type_menu.grid(row=1, column=1, padx=2, pady=2)
        ttk.Label(self.parent, text="Function Type:").grid(row=1, column=0, padx=2, pady=2)

        # Fetch initial function configuration and parameters
        self._fetch_noise_function(self.ticker_var.get())

        # Save button
        save_button = ttk.Button(self.parent, text="Save Noise", command=self.save_noise)
        save_button.grid(row=50, column=0, columnspan=2, padx=2, pady=2)

    def save_noise(self):
        ticker = self.ticker_var.get()
        function_type = self.function_type_var.get()
        noise_params = {NoiseFunction.get_parameter_spec(function_type)[i]: entry.get() for i, entry in enumerate(self.parameter_entries)}
        query = self.session.prepare("INSERT INTO shares.noise_functions (share_ticker, noise_function_type, noise_parameters) VALUES (?, ?, ?);")
        try:
            self.session.execute(query, (ticker, function_type, noise_params))
            messagebox.showinfo("Success", "Noise has been saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save noise. Error: {e}")

        for entry in self.parameter_entries:
            entry.delete(0, 'end')



class NewLinkage:
    def __init__(self, parent, session):
        self.parent = parent
        self.session = session
        self.tickers = self.get_all_tickers()
        self.function_types = LinkageFunction.get_linkage_function_types()
        self.param_widgets = []
        self.param_labels = []
        self.create_widgets()

    def get_all_tickers(self):
        rows = self.session.execute('SELECT share_ticker FROM shares.share_details;')
        tickers = [row.share_ticker for row in rows]
        tickers.sort()
        return tickers

    def create_widgets(self):
        self.primary_ticker_var = tk.StringVar(self.parent)
        self.primary_ticker_var.set(self.tickers[0])
        # Explicitly pass the initial value and then the list of options
        primary_ticker_menu = ttk.OptionMenu(self.parent, self.primary_ticker_var, self.tickers[0], *self.tickers)
        ttk.Label(self.parent, text="Primary Ticker:").grid(row=0, column=0, padx=2, pady=2)
        primary_ticker_menu.grid(row=0, column=1, padx=2, pady=2)
        
        self.secondary_ticker_var = tk.StringVar(self.parent)
        self.secondary_ticker_var.set(self.tickers[0])
        # Similarly, for the secondary ticker
        secondary_ticker_menu = ttk.OptionMenu(self.parent, self.secondary_ticker_var, self.tickers[0], *self.tickers)
        secondary_ticker_menu.grid(row=0, column=5, padx=2, pady=2)
        ttk.Label(self.parent, text="Secondary Ticker:").grid(row=0, column=4, padx=2, pady=2)

        self.function_type_var = tk.StringVar(self.parent)
        self.function_type_var.set(self.function_types[0])
        self.function_type_var.trace('w', self.update_param_fields)
        # And for the function type
        function_type_menu = ttk.OptionMenu(self.parent, self.function_type_var, self.function_types[0], *self.function_types)
        ttk.Label(self.parent, text="Function Type:").grid(row=1, column=0, padx=2, pady=2)
        function_type_menu.grid(row=1, column=1, padx=2, pady=2)

        self.update_param_fields()

        save_button = ttk.Button(self.parent, text="Save Linkage", command=self.save_linkage)
        save_button.grid(row=10, column=0, columnspan=2, padx=2, pady=2)


    def update_param_fields(self, *args):
        for widget in self.param_widgets:
            widget.destroy()
        for label in self.param_labels:
            label.destroy()

        self.param_widgets = []
        self.param_labels = []
        function_type = self.function_type_var.get()
        params = LinkageFunction.get_parameter_spec(function_type)
        for i, param in enumerate(params, start=2):
            label = ttk.Label(self.parent, text=f"{param}:")
            label.grid(row=i, column=0, padx=1, pady=1)
            self.param_labels.append(label)

            entry = ttk.Entry(self.parent)
            entry.grid(row=i, column=1, padx=1, pady=1)
            self.param_widgets.append(entry)
        
    def save_linkage(self):
        linkage_id = uuid4()
        primary_ticker = self.primary_ticker_var.get()
        secondary_ticker = self.secondary_ticker_var.get()
        function_type = self.function_type_var.get()

        linkage_parameters = {param: entry.get() for param, entry in zip(LinkageFunction.get_parameter_spec(function_type), self.param_widgets)}

        query = self.session.prepare("INSERT INTO shares.linkage_functions (primary_share_ticker, secondary_share_ticker, linkage_id, linkage_function_type, linkage_parameters) VALUES (?, ?, ?, ?, ?);")
        try:
            self.session.execute(query, (primary_ticker, secondary_ticker, linkage_id, function_type, linkage_parameters))
            messagebox.showinfo("Success", "Share linkage has been saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save share linkage. Error: {e}")

        for entry in self.param_widgets:
            entry.delete(0, 'end')
        for label in self.param_labels:
            label.config(text="")



def get_all_linkages(session):
    linkages = []
    rows = session.execute("SELECT * FROM shares.linkage_functions")
    for row in rows:
        linkage = {
            'linkage_id': row.linkage_id,
            'primary_share_ticker': row.primary_share_ticker,
            'secondary_share_ticker': row.secondary_share_ticker,
            'linkage_function_type': row.linkage_function_type,
            'linkage_parameters': row.linkage_parameters,
        }
        linkages.append(linkage)
    linkages = sorted(linkages, key=lambda linkage: linkage['primary_share_ticker'])
    return linkages



class LinkageEditor:
    def __init__(self, parent, session):
        self.parent = parent
        self.session = session
        self.function_types = LinkageFunction.get_linkage_function_types()
        self.linkages = get_all_linkages(self.session)
        self.create_widgets()

    def create_widgets(self):
        self.linkage_vars = []
        for i, linkage in enumerate(self.linkages):
            ttk.Label(self.parent, text=f"Linkage ID: {linkage['linkage_id']}").grid(row=i, column=0, padx=1, pady=1)
            ttk.Label(self.parent, text=f"Primary Ticker: {linkage['primary_share_ticker']}").grid(row=i, column=1, padx=1, pady=1)
            ttk.Label(self.parent, text=f"Secondary Ticker: {linkage['secondary_share_ticker']}").grid(row=i, column=2, padx=1, pady=1)

            function_type_var = tk.StringVar(self.parent)
            function_type_menu = ttk.OptionMenu(self.parent, function_type_var, linkage['linkage_function_type'], *self.function_types)
            function_type_menu.grid(row=i, column=3, padx=1, pady=1)
            function_type_var.set(linkage['linkage_function_type'])

            self.linkage_vars.append((linkage['linkage_id'], function_type_var, linkage['linkage_parameters']))

            delete_button = ttk.Button(self.parent, text="Delete", command=lambda linkage_id=linkage['linkage_id']: self.delete_linkage(linkage_id))
            delete_button.grid(row=i, column=14, padx=1, pady=1)

            self.parent.after_idle(function_type_var.trace, 'w', lambda *args, i=i, function_type_var=function_type_var, linkage_params=linkage['linkage_parameters']: self.update_param_fields(i, function_type_var.get(), linkage_params))

        update_button = ttk.Button(self.parent, text="Update Linkages", command=self.update_linkages)
        update_button.grid(row=len(self.linkages) + 1, column=0, columnspan=8, padx=1, pady=1)

        for i, linkage in enumerate(self.linkages):
            self.update_param_fields(i, linkage['linkage_function_type'], linkage['linkage_parameters'])



    def delete_linkage(self, linkage_id):
        query = self.session.prepare("DELETE FROM shares.linkage_functions WHERE linkage_id = ?")
        try:
            self.session.execute(query, (linkage_id,))
            messagebox.showinfo("Success", f"Linkage {linkage_id} has been deleted.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete linkage {linkage_id}. Error: {e}")

        self.linkages = get_all_linkages(self.session)
        for widget in self.parent.winfo_children():
            widget.destroy()
        self.create_widgets()

    def update_param_fields(self, i, function_type, linkage_params):
        for widget in getattr(self, f"param_widgets_{i}", []):
            widget.destroy()
        for label in getattr(self, f"param_labels_{i}", []):
            label.destroy()

        self.param_widgets = []
        self.param_labels = []

        params = LinkageFunction.get_parameter_spec(function_type)
        for j, param in enumerate(params, start=1):
            label = ttk.Label(self.parent, text=f"{param}:")
            label.grid(row=i, column=4+j*2, padx=1, pady=1)
            self.param_labels.append(label)

            entry = ttk.Entry(self.parent)
            entry.insert(0, linkage_params.get(param, ''))
            entry.grid(row=i, column=5+j*2, padx=1, pady=1)
            self.param_widgets.append(entry)

        setattr(self, f"param_widgets_{i}", self.param_widgets)
        setattr(self, f"param_labels_{i}", self.param_labels)

    def update_linkages_old(self):
        for i, (linkage_id, function_type_var, old_params) in enumerate(self.linkage_vars):
            function_type = function_type_var.get()

            linkage_params = {param: widget.get() for param, widget in zip(LinkageFunction.get_parameter_spec(function_type), getattr(self, f"param_widgets_{i}"))}

            if old_params != linkage_params or old_params['linkage_function_type'] != function_type:
                query = self.session.prepare("UPDATE shares.linkage_functions SET linkage_function_type = ?, linkage_parameters = ? WHERE linkage_id = ?")
                try:
                    self.session.execute(query, (function_type, linkage_params, linkage_id))
                    messagebox.showinfo("Success", f"Linkage {linkage_id} has been updated.")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to update linkage {linkage_id}. Error: {e}")

    def update_linkages(self):
        for i, (linkage_id, function_type_var, old_params) in enumerate(self.linkage_vars):
            function_type = function_type_var.get()

            # Collect new parameters from entry widgets
            linkage_params = {
                param: widget.get() for param, widget in zip(
                    LinkageFunction.get_parameter_spec(function_type), getattr(self, f"param_widgets_{i}")
                )
            }

            # Check for existence of 'linkage_function_type' in old_params before comparing
            old_function_type = old_params.get('linkage_function_type')
            if old_params != linkage_params or old_function_type != function_type:
                query = self.session.prepare(
                    "UPDATE shares.linkage_functions SET linkage_function_type = ?, linkage_parameters = ? WHERE linkage_id = ?"
                )
                try:
                    self.session.execute(query, (function_type, linkage_params, linkage_id))
                    messagebox.showinfo("Success", f"Linkage {linkage_id} has been updated.")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to update linkage {linkage_id}. Error: {e}")

def get_all_shares(session):
    shares = []
    rows = session.execute("SELECT * FROM shares.share_details")
    for row in rows:
        share = {
            'share_ticker': row.share_ticker,
            'share_name': row.share_name,
            'share_description': row.share_description,
            'share_value': row.share_value,
            'share_events': row.share_events,
        }
        shares.append(share)

    shares = sorted(shares, key=lambda share: share['share_ticker'])
    return shares



class ShareEditor:
    def __init__(self, parent, session):
        self.parent = parent
        self.session = session
        self.shares = get_all_shares(self.session)
        self.entries = {}
        self.share_limit_per_row = 6

        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill='both', expand=True)

        self.scrollbar = ttk.Scrollbar(self.main_frame, orient='vertical')
        self.scrollbar.pack(side='right', fill='y', expand=False)

        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(side='left', fill='both', expand=True)

        self.canvas = tk.Canvas(self.content_frame, borderwidth=0, highlightthickness=0)
        self.canvas.pack(side='left', fill='both', expand=True)

        self.scrollable_frame = ttk.Frame(self.canvas)
        self.window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.configure(command=self.canvas.yview)

        self.canvas.bind("<Configure>", lambda e: self.parent.after_idle(self.update_scrollregion))

        self.main_frame.bind_all('<MouseWheel>', lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        self.create_cards()
        self.parent.after_idle(self.update_scrollregion)  # Schedule an initial update

    def update_scrollregion(self):
        self.canvas.update_idletasks()
        frame_width = self.scrollable_frame.winfo_reqwidth()
        canvas_width = self.canvas.winfo_width()
        x1 = max(0, (canvas_width - frame_width) // 2)
        self.canvas.config(scrollregion=(0, 0, frame_width, self.scrollable_frame.winfo_height()))
        self.canvas.coords(self.window, (x1, 0))

    def create_cards(self):
        row_frame = None  # Initialize outside the loop to keep scope
        self.shares = get_all_shares(self.session)

        for i, share in enumerate(self.shares):
            if i % self.share_limit_per_row == 0:
                row_frame = ttk.Frame(self.scrollable_frame)
                row_frame.pack(pady=10, anchor='center')  # Center the row frame

            card_frame = ttk.Frame(row_frame, borderwidth=2, relief="groove", padding=10)
            # Center cards by dynamically calculating the column span
            column_span = (self.share_limit_per_row - len(self.shares) % self.share_limit_per_row) if (i // self.share_limit_per_row == len(self.shares) // self.share_limit_per_row) else 0
            card_frame.grid(row=0, column=i % self.share_limit_per_row, columnspan=max(1, column_span), padx=10, pady=10)

            ttk.Label(card_frame, text=f"Share Ticker: {share['share_ticker']}").pack(anchor='w')
            for key, value in share.items():
                if key != 'share_ticker':
                    row = ttk.Frame(card_frame)
                    row.pack(fill='x', expand=True, pady=2)
                    ttk.Label(row, text=f"{key}:").pack(side='left')
                    entry = ttk.Entry(row)
                    entry.insert(0, str(value))
                    entry.pack(side='right', expand=False)
                    self.entries[(share['share_ticker'], key)] = entry

            save_button = ttk.Button(card_frame, text="Save Changes", command=lambda share_ticker=share['share_ticker']: self.save_changes_single(share_ticker))
            save_button.pack(fill='x', expand=True, pady=2)
            delete_button = ttk.Button(card_frame, text="Delete Share", command=lambda share_ticker=share['share_ticker']: self.delete_share(share_ticker))
            delete_button.pack(fill='x', expand=True, pady=2)




    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def refresh_view(self):
        self.clear_frame(self.scrollable_frame)
        self.create_cards()

    def delete_share(self, share_ticker):
        # Prepare queries for finding IDs and deleting entries with ALLOW FILTERING
        find_ids_primary = self.session.prepare(
            "SELECT linkage_id FROM shares.linkage_functions WHERE primary_share_ticker = ? ALLOW FILTERING")
        find_ids_secondary = self.session.prepare(
            "SELECT linkage_id FROM shares.linkage_functions WHERE secondary_share_ticker = ? ALLOW FILTERING")
        delete_by_id = self.session.prepare(
            "DELETE FROM shares.linkage_functions WHERE linkage_id = ?")
        
        del_noise = self.session.prepare(
            "DELETE FROM shares.noise_functions WHERE share_ticker = ?")
        del_price_history = self.session.prepare(
            "DELETE FROM shares.price_history WHERE share_ticker = ?")
        del_share_details = self.session.prepare(
            "DELETE FROM shares.share_details WHERE share_ticker = ?")

        try:
            # Collect linkage_ids to delete
            linkage_ids_to_delete = set()
            for row in self.session.execute(find_ids_primary, (share_ticker,)):
                linkage_ids_to_delete.add(row.linkage_id)
            for row in self.session.execute(find_ids_secondary, (share_ticker,)):
                linkage_ids_to_delete.add(row.linkage_id)

            # Execute deletion for each collected linkage_id
            for linkage_id in linkage_ids_to_delete:
                self.session.execute(delete_by_id, (linkage_id,))

            # Execute other deletions
            self.session.execute(del_noise, (share_ticker,))
            self.session.execute(del_price_history, (share_ticker,))
            self.session.execute(del_share_details, (share_ticker,))

            # Success message
            messagebox.showinfo("Success", f"All records related to share {share_ticker} have been deleted.")
        except Exception as e:
            # Error handling
            messagebox.showerror("Error", f"Failed to delete all records related to share {share_ticker}. Error: {e}")

        self.refresh_view()


    def save_changes_single(self, share_ticker):
        share = [share for share in self.shares if share['share_ticker'] == share_ticker][0]
        for key in share:
            if key == 'share_ticker':
                continue
            entry = self.entries[(share['share_ticker'], key)]
            value = entry.get()
            if key == 'share_value':
                value = float(value)
            if key == 'share_events':
                value = {}
            
            query = self.session.prepare(f"UPDATE shares.share_details SET {key} = ? WHERE share_ticker = ?")
            try:
                self.session.execute(query, (value, share['share_ticker']))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update share {share['share_ticker']}. Error: {e}")
            
        self.refresh_view()
    


def plot_price_history_candlesticks(ticker, session):
    query = "SELECT * FROM shares.price_history WHERE share_ticker = %s"
    results = session.execute(query, (ticker,))
    df = pd.DataFrame(list(results))
    
    df.rename(columns={'value': 'Adj Close'}, inplace=True)

    df.set_index('timestamp', inplace=True)
    df.index.name = 'Date'

    df['Open'] = df['Adj Close'].shift(1) 
    df['High'] = df[['Open', 'Adj Close']].max(axis=1)
    df['Low'] = df[['Open', 'Adj Close']].min(axis=1)
    df['Close'] = df['Adj Close']

    df_ohlc = df[['Open', 'High', 'Low', 'Close']].reset_index()
    df_ohlc['Date'] = df_ohlc['Date'].apply(mdates.date2num)

    fig, ax = plt.subplots(figsize=(14,7))
    candlestick_ohlc(ax, df_ohlc.values, width=0.6, colorup='g', colordown='r', alpha=0.8)

    ax.xaxis_date()  
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.title(f'{ticker} Adjusted Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.grid(True)
    plt.show()



def plot_two_price_histories(ticker1, ticker2, session):
    fig, ax = plt.subplots(figsize=(14,7))
    
    for i, ticker in enumerate([ticker1, ticker2]):
        query = "SELECT * FROM shares.price_history WHERE share_ticker = %s"
        results = session.execute(query, (ticker,))
        df = pd.DataFrame(list(results))
        print(f"Length of {ticker} data: {len(df)}")
        print ("##################################################")
        df.rename(columns={'value': 'Adj Close'}, inplace=True)

        df.set_index('timestamp', inplace=True)
        df.index.name = 'Date'

        if i == 0:
            df['Open'] = df['Adj Close'].shift(1) 
            df['High'] = df[['Open', 'Adj Close']].max(axis=1)
            df['Low'] = df[['Open', 'Adj Close']].min(axis=1)
            df['Close'] = df['Adj Close']

            df_ohlc = df[['Open', 'High', 'Low', 'Close']].reset_index()
            df_ohlc['Date'] = df_ohlc['Date'].apply(mdates.date2num)

            candlestick_ohlc(ax, df_ohlc.values, width=0.6, colorup='g', colordown='r', alpha=0.8)

        else:
            ax.plot(df.index, df['Adj Close'], label=ticker)
    
    ax.xaxis_date()  
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.title(f'{ticker1} and {ticker2} Adjusted Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_multiple_price_histories(ticker1, tickers, session):
    fig, ax = plt.subplots(figsize=(14,7))

    query = "SELECT * FROM shares.price_history WHERE share_ticker = %s"
    results = session.execute(query, (ticker1,))
    df = pd.DataFrame(list(results))

    df.rename(columns={'value': 'Adj Close'}, inplace=True)

    df.set_index('timestamp', inplace=True)
    df.index.name = 'Date'

    df['Open'] = df['Adj Close'].shift(1) 
    df['High'] = df[['Open', 'Adj Close']].max(axis=1)
    df['Low'] = df[['Open', 'Adj Close']].min(axis=1)
    df['Close'] = df['Adj Close']

    df_ohlc = df[['Open', 'High', 'Low', 'Close']].reset_index()
    df_ohlc['Date'] = df_ohlc['Date'].apply(mdates.date2num)

    candlestick_ohlc(ax, df_ohlc.values, width=0.6, colorup='g', colordown='r', alpha=0.8)

    for ticker in tickers:
        query = "SELECT * FROM shares.price_history WHERE share_ticker = %s"
        results = session.execute(query, (ticker,))
        df = pd.DataFrame(list(results))

        df.rename(columns={'value': 'Adj Close'}, inplace=True)

        df.set_index('timestamp', inplace=True)
        df.index.name = 'Date'

        ax.plot(df.index, df['Adj Close'], label=ticker)

    ax.xaxis_date()  
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.title(f'{ticker1} and other tickers Adjusted Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()



class SharePlotTabOld:
    def __init__(self, parent, session):
        self.parent = parent
        self.session = session
        self.shares = get_all_shares(self.session)
        
        self.button_frame = ttk.Frame(self.parent)
        self.button_frame.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)

        self.canvas = tk.Canvas(self.button_frame)
        self.scrollbar = ttk.Scrollbar(self.button_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.share_frame = ttk.Frame(self.canvas)
        self.share_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.share_frame, anchor="nw")

        # Create sub-frames for each category
        self.create_fictional_share_buttons(self.share_frame)
        self.create_GICS_share_buttons(self.share_frame)
        self.create_real_share_buttons(self.share_frame)

        self.canvas.pack(side='left', fill='both', expand=True)
        self.scrollbar.pack(side='right', fill='y')

        self.plot_frame = None
        self.parent.grid_columnconfigure(0, weight=1)
        self.parent.grid_rowconfigure(0, weight=1)

        # Configure plot frame to take the full height on the right
        self.parent.grid_rowconfigure(0, weight=1)
        self.parent.grid_columnconfigure(1, weight=4)

    def create_share_buttons(self, frame, shares):
        for i, share in enumerate(shares):
            button = ttk.Button(frame, text=share['share_ticker'], command=lambda share=share: self.plot_share(share))
            button.pack(fill='x', padx=2, pady=2)

    def create_fictional_share_buttons(self, frame):
        fictional_frame = ttk.Frame(frame)
        fictional_frame.pack(side='left', fill='both', expand=True)
        fictional_shares = [share for share in self.shares if share['share_value'] > 0]
        self.create_share_buttons(fictional_frame, fictional_shares)

    def create_GICS_share_buttons(self, frame):
        GICS_frame = ttk.Frame(frame)
        GICS_frame.pack(side='left', fill='both', expand=True)
        GICS_shares = [share for share in self.shares if share['share_value'] < 0 and share['share_ticker'].startswith('XL')]
        self.create_share_buttons(GICS_frame, GICS_shares)

    def create_real_share_buttons(self, frame):
        real_frame = ttk.Frame(frame)
        real_frame.pack(side='left', fill='both', expand=True)
        real_shares = [share for share in self.shares if share['share_value'] < 0 and not share['share_ticker'].startswith('XL')]
        self.create_share_buttons(real_frame, real_shares)

    def plot_share(self, share):
        if self.plot_frame is not None:
            self.plot_frame.destroy()

        self.plot_frame = ttk.Frame(self.parent)
        self.plot_frame.grid(row=0, column=1, sticky='nsew')
        self.parent.grid_columnconfigure(1, weight=3)

        ticker = share['share_ticker']

        query = "SELECT * FROM shares.price_history WHERE share_ticker = %s"
        results = self.session.execute(query, (ticker,))
        df = pd.DataFrame(list(results))
        
        df.rename(columns={'value': 'Adj Close'}, inplace=True)
        df.set_index('timestamp', inplace=True)
        df.index.name = 'Date'

        df['Open'] = df['Adj Close'].shift(1)
        df['High'] = df[['Open', 'Adj Close']].max(axis=1)
        df['Low'] = df[['Open', 'Adj Close']].min(axis=1)
        df['Close'] = df['Adj Close']

        df_ohlc = df[['Open', 'High', 'Low', 'Close']].reset_index()
        df_ohlc['Date'] = df_ohlc['Date'].apply(mdates.date2num)

        fig = plt.Figure(figsize=(15, 5), dpi=100)
        ax = fig.add_subplot(111)
        candlestick_ohlc(ax, df_ohlc.values, width=0.6, colorup='g', colordown='r', alpha=0.8)

        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.set_title(f'{ticker} Adjusted Close Price Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Adjusted Close Price')
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

class FileProcessingTab:
    def __init__(self, parent, session):
        self.parent = parent
        self.session = session
        self.create_widgets()

    def create_widgets(self):
        file_button = ttk.Button(self.parent, text="Process File", command=self.file_button_clicked)
        file_button.pack(padx=2, pady=2)

        folder_button = ttk.Button(self.parent, text="Process Folder", command=self.folder_button_clicked)
        folder_button.pack(padx=2, pady=2)

    def file_button_clicked(self):
        file_path = filedialog.askopenfilename(filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
        if file_path:
            process_file(file_path, self.session)

    def folder_button_clicked(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            process_all_files(dir_path, self.session)



class SharePlotTab:
    def __init__(self, parent, session):
        self.parent = parent
        self.session = session
        self.shares = get_all_shares(self.session)
        
        self.button_frame = ttk.Frame(self.parent)
        self.button_frame.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        
        self.canvas = tk.Canvas(self.button_frame)
        self.scrollbar = ttk.Scrollbar(self.button_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.scrollbar.set)

        self.share_frame = ttk.Frame(self.canvas)
        self.share_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.share_frame, anchor="nw")

        # Create sub-frames for each category in different rows
        self.create_fictional_share_buttons(self.share_frame)
        self.create_GICS_share_buttons(self.share_frame)
        self.create_real_share_buttons(self.share_frame)

        self.canvas.pack(side='top', fill='both', expand=True)
        self.scrollbar.pack(side='bottom', fill='x')

        self.plot_frame = ttk.Frame(self.parent)
        self.plot_frame.grid(row=1, column=0, sticky='nsew')
        self.parent.grid_rowconfigure(1, weight=1)
        self.parent.grid_columnconfigure(0, weight=1)

    def create_label(self, frame, text):
        label = ttk.Label(frame, text=text)
        label.grid(row=0, column=0, sticky='w', padx=2, pady=2)

    def create_share_buttons(self, frame, shares):
        for i, share in enumerate(shares):
            button = ttk.Button(frame, text=share['share_ticker'], command=lambda share=share: self.plot_share(share))
            button.grid(row=1, column=i, sticky='ew', padx=2, pady=2)

    def create_fictional_share_buttons(self, frame):
        fictional_frame = ttk.Frame(frame)
        fictional_frame.grid(row=0, column=0, sticky='nsew')
        self.create_label(fictional_frame, "Fictional Shares")
        fictional_shares = [share for share in self.shares if share['share_value'] > 0]
        self.create_share_buttons(fictional_frame, fictional_shares)

    def create_GICS_share_buttons(self, frame):
        GICS_frame = ttk.Frame(frame)
        GICS_frame.grid(row=1, column=0, sticky='nsew')
        self.create_label(GICS_frame, "GICS Shares")
        GICS_shares = [share for share in self.shares if share['share_value'] < 0 and share['share_ticker'].startswith('XL')]
        self.create_share_buttons(GICS_frame, GICS_shares)

    def create_real_share_buttons(self, frame):
        real_frame = ttk.Frame(frame)
        real_frame.grid(row=2, column=0, sticky='nsew')
        self.create_label(real_frame, "Real Shares")
        real_shares = [share for share in self.shares if share['share_value'] < 0 and not share['share_ticker'].startswith('XL')]
        self.create_share_buttons(real_frame, real_shares)



    def plot_share(self, share):
        if self.plot_frame is not None:
            self.plot_frame.destroy()

        self.plot_frame = ttk.Frame(self.parent)
        self.plot_frame.grid(row=1, column=0, sticky='nsew')
        self.parent.grid_columnconfigure(0, weight=3)

        ticker = share['share_ticker']

        query = "SELECT * FROM shares.price_history WHERE share_ticker = %s"
        results = self.session.execute(query, (ticker,))
        df = pd.DataFrame(list(results))
        
        df.rename(columns={'value': 'Adj Close'}, inplace=True)
        df.set_index('timestamp', inplace=True)
        df.index.name = 'Date'

        df['Open'] = df['Adj Close'].shift(1)
        df['High'] = df[['Open', 'Adj Close']].max(axis=1)
        df['Low'] = df[['Open', 'Adj Close']].min(axis=1)
        df['Close'] = df['Adj Close']

        df_ohlc = df[['Open', 'High', 'Low', 'Close']].reset_index()
        df_ohlc['Date'] = df_ohlc['Date'].apply(mdates.date2num)

        fig = plt.Figure(figsize=(15, 5), dpi=100)
        ax = fig.add_subplot(111)
        candlestick_ohlc(ax, df_ohlc.values, width=0.6, colorup='g', colordown='r', alpha=0.8)

        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.set_title(f'{ticker} Adjusted Close Price Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Adjusted Close Price')
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)



class TabView(ttk.Window):
    def __init__(self, session, cluster, tab_names, *tabs, theme = 'sandstone'):
        super().__init__(themename=theme)
        self.title("Share Data")
        self.session = session
        self.cluster = cluster
        self.notebook = ttk.Notebook(self)
        self.tabs = tabs
        self.tab_names = tab_names
        self.state('zoomed')

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.create_tabs()

    def create_tabs(self):
        for i, (tab_name, tab_func) in enumerate(zip(self.tab_names, self.tabs)):
            tab_frame = ttk.Frame(self.notebook, padding=25)
            self.notebook.add(tab_frame, text=tab_name)
            tab_func(tab_frame)

        self.notebook.pack(expand=True, fill='both')

    def on_close(self):
        print("Closing Cassandra session...")
        self.session.shutdown()
        self.cluster.shutdown()
        self.destroy()
        sys.exit()



def delete_shares_starting_with_number(session):
    # Prepare the query to find all share tickers starting with a number
    find_shares = session.prepare("SELECT share_ticker FROM shares.share_details WHERE share_ticker >= '0' AND share_ticker < ':' ALLOW FILTERING")

    # Collect all share tickers that start with a number
    shares_to_delete = session.execute(find_shares)

    # Function to handle the deep delete operation for each ticker
    def deep_delete_share(share_ticker):
        # Fetching linkage_ids for deletion
        find_ids_primary = session.prepare(
            "SELECT linkage_id FROM shares.linkage_functions WHERE primary_share_ticker = ? ALLOW FILTERING")
        find_ids_secondary = session.prepare(
            "SELECT linkage_id FROM shares.linkage_functions WHERE secondary_share_ticker = ? ALLOW FILTERING")
        delete_by_id = session.prepare(
            "DELETE FROM shares.linkage_functions WHERE linkage_id = ?")
        
        del_noise = session.prepare(
            "DELETE FROM shares.noise_functions WHERE share_ticker = ?")
        del_price_history = session.prepare(
            "DELETE FROM shares.price_history WHERE share_ticker = ?")
        del_share_details = session.prepare(
            "DELETE FROM shares.share_details WHERE share_ticker = ?")

        try:
            # Collect linkage_ids for deletion
            linkage_ids_to_delete = set()
            primary_ids = session.execute(find_ids_primary, (share_ticker,))
            secondary_ids = session.execute(find_ids_secondary, (share_ticker,))
            for row in primary_ids:
                linkage_ids_to_delete.add(row.linkage_id)
            for row in secondary_ids:
                linkage_ids_to_delete.add(row.linkage_id)

            # Execute deletion for each collected linkage_id
            for linkage_id in linkage_ids_to_delete:
                session.execute(delete_by_id, (linkage_id,))

            # Execute other deletions
            session.execute(del_noise, (share_ticker,))
            session.execute(del_price_history, (share_ticker,))
            session.execute(del_share_details, (share_ticker,))

            print(f"Successfully deleted all records related to share {share_ticker}.")
        except Exception as e:
            print(f"Failed to delete records related to share {share_ticker}. Error: {e}")

    # Loop through and delete each ticker
    for row in shares_to_delete:
        deep_delete_share(row.share_ticker)


#delete
if __name__ == "__main_":
    session, cluster = get_cassandra_session()
    delete_shares_starting_with_number(session)
    cluster.shutdown()


if __name__ == "__main__":
    session, cluster = get_cassandra_session()

    if(session.execute("SELECT * FROM shares.share_details").one() == None):
        tab_names = ["New Share"]
        TabView(session, cluster, tab_names, lambda parent: NewShare(parent, session)).mainloop()
    else:
        tab_names = ["New Share", "Edit Shares", "Noises", "New Linkage", "Linkage Editor", "Plot or generate", "Import Data", "Settings"]
        TabView(session, cluster, tab_names, 
                        lambda parent: NewShare(parent, session),
                        lambda parent: ShareEditor(parent, session),
                        lambda parent: AddNoise(parent, session),
                        lambda parent: NewLinkage(parent, session),
                        lambda parent: LinkageEditor(parent, session),
                        lambda parent: SharePlotTab(parent, session),
                        lambda parent: FileProcessingTab(parent, session)
                        ).mainloop()
