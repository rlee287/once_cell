//! Thread-safe, non-blocking, "first one wins" flavor of `OnceCell`.
//!
//! If two threads race to initialize a type from the `race` module, they
//! don't block, execute initialization function together, but only one of
//! them stores the result.
//!
//! This module does not require `std` feature.
//!
//! # Atomic orderings
//!
//! All types in this module use `Acquire` and `Release`
//! [atomic orderings](Ordering) for all their operations. While this is not
//! strictly necessary for types other than `OnceBox`, it is useful for users as
//! it allows them to be certain that after `get` or `get_or_init` returns on
//! one thread, any side-effects caused by the setter thread prior to them
//! calling `set` or `get_or_init` will be made visible to that thread; without
//! it, it's possible for it to appear as if they haven't happened yet from the
//! getter thread's perspective. This is an acceptable tradeoff to make since
//! `Acquire` and `Release` have very little performance overhead on most
//! architectures versus `Relaxed`.

#[cfg(feature = "critical-section")]
use portable_atomic as atomic;
#[cfg(not(feature = "critical-section"))]
use core::sync::atomic;

use atomic::{AtomicPtr, AtomicUsize, Ordering};
use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::num::NonZeroUsize;
use core::ptr;

/// A thread-safe cell which can be written to only once.
#[derive(Default, Debug)]
pub struct OnceNonZeroUsize {
    inner: AtomicUsize,
}

impl OnceNonZeroUsize {
    /// Creates a new empty cell.
    #[inline]
    pub const fn new() -> OnceNonZeroUsize {
        OnceNonZeroUsize { inner: AtomicUsize::new(0) }
    }

    /// Gets the underlying value.
    #[inline]
    pub fn get(&self) -> Option<NonZeroUsize> {
        let val = self.inner.load(Ordering::Acquire);
        NonZeroUsize::new(val)
    }

    /// Sets the contents of this cell to `value`.
    ///
    /// Returns `Ok(())` if the cell was empty and `Err(())` if it was
    /// full.
    #[inline]
    pub fn set(&self, value: NonZeroUsize) -> Result<(), ()> {
        let exchange =
            self.inner.compare_exchange(0, value.get(), Ordering::AcqRel, Ordering::Acquire);
        match exchange {
            Ok(_) => Ok(()),
            Err(_) => Err(()),
        }
    }

    /// Gets the contents of the cell, initializing it with `f` if the cell was
    /// empty.
    ///
    /// If several threads concurrently run `get_or_init`, more than one `f` can
    /// be called. However, all threads will return the same value, produced by
    /// some `f`.
    pub fn get_or_init<F>(&self, f: F) -> NonZeroUsize
    where
        F: FnOnce() -> NonZeroUsize,
    {
        enum Void {}
        match self.get_or_try_init(|| Ok::<NonZeroUsize, Void>(f())) {
            Ok(val) => val,
            Err(void) => match void {},
        }
    }

    /// Gets the contents of the cell, initializing it with `f` if
    /// the cell was empty. If the cell was empty and `f` failed, an
    /// error is returned.
    ///
    /// If several threads concurrently run `get_or_init`, more than one `f` can
    /// be called. However, all threads will return the same value, produced by
    /// some `f`.
    pub fn get_or_try_init<F, E>(&self, f: F) -> Result<NonZeroUsize, E>
    where
        F: FnOnce() -> Result<NonZeroUsize, E>,
    {
        let val = self.inner.load(Ordering::Acquire);
        let res = match NonZeroUsize::new(val) {
            Some(it) => it,
            None => {
                let mut val = f()?.get();
                let exchange =
                    self.inner.compare_exchange(0, val, Ordering::AcqRel, Ordering::Acquire);
                if let Err(old) = exchange {
                    val = old;
                }
                unsafe { NonZeroUsize::new_unchecked(val) }
            }
        };
        Ok(res)
    }
}

/// A thread-safe cell which can be written to only once.
#[derive(Default, Debug)]
pub struct OnceBool {
    inner: OnceNonZeroUsize,
}

impl OnceBool {
    /// Creates a new empty cell.
    #[inline]
    pub const fn new() -> OnceBool {
        OnceBool { inner: OnceNonZeroUsize::new() }
    }

    /// Gets the underlying value.
    #[inline]
    pub fn get(&self) -> Option<bool> {
        self.inner.get().map(OnceBool::from_usize)
    }

    /// Sets the contents of this cell to `value`.
    ///
    /// Returns `Ok(())` if the cell was empty and `Err(())` if it was
    /// full.
    #[inline]
    pub fn set(&self, value: bool) -> Result<(), ()> {
        self.inner.set(OnceBool::to_usize(value))
    }

    /// Gets the contents of the cell, initializing it with `f` if the cell was
    /// empty.
    ///
    /// If several threads concurrently run `get_or_init`, more than one `f` can
    /// be called. However, all threads will return the same value, produced by
    /// some `f`.
    pub fn get_or_init<F>(&self, f: F) -> bool
    where
        F: FnOnce() -> bool,
    {
        OnceBool::from_usize(self.inner.get_or_init(|| OnceBool::to_usize(f())))
    }

    /// Gets the contents of the cell, initializing it with `f` if
    /// the cell was empty. If the cell was empty and `f` failed, an
    /// error is returned.
    ///
    /// If several threads concurrently run `get_or_init`, more than one `f` can
    /// be called. However, all threads will return the same value, produced by
    /// some `f`.
    pub fn get_or_try_init<F, E>(&self, f: F) -> Result<bool, E>
    where
        F: FnOnce() -> Result<bool, E>,
    {
        self.inner.get_or_try_init(|| f().map(OnceBool::to_usize)).map(OnceBool::from_usize)
    }

    #[inline]
    fn from_usize(value: NonZeroUsize) -> bool {
        value.get() == 1
    }

    #[inline]
    fn to_usize(value: bool) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(if value { 1 } else { 2 }) }
    }
}

/// A thread-safe cell which can be written to only once.
pub struct OnceRef<'a, T> {
    inner: AtomicPtr<T>,
    ghost: PhantomData<UnsafeCell<&'a T>>,
}

// TODO: Replace UnsafeCell with SyncUnsafeCell once stabilized
unsafe impl<'a, T: Sync> Sync for OnceRef<'a, T> {}

impl<'a, T> core::fmt::Debug for OnceRef<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "OnceRef({:?})", self.inner)
    }
}

impl<'a, T> Default for OnceRef<'a, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T> OnceRef<'a, T> {
    /// Creates a new empty cell.
    pub const fn new() -> OnceRef<'a, T> {
        OnceRef { inner: AtomicPtr::new(ptr::null_mut()), ghost: PhantomData }
    }

    /// Gets a reference to the underlying value.
    pub fn get(&self) -> Option<&'a T> {
        let ptr = self.inner.load(Ordering::Acquire);
        unsafe { ptr.as_ref() }
    }

    /// Sets the contents of this cell to `value`.
    ///
    /// Returns `Ok(())` if the cell was empty and `Err(value)` if it was
    /// full.
    pub fn set(&self, value: &'a T) -> Result<(), ()> {
        let ptr = value as *const T as *mut T;
        let exchange =
            self.inner.compare_exchange(ptr::null_mut(), ptr, Ordering::AcqRel, Ordering::Acquire);
        match exchange {
            Ok(_) => Ok(()),
            Err(_) => Err(()),
        }
    }

    /// Gets the contents of the cell, initializing it with `f` if the cell was
    /// empty.
    ///
    /// If several threads concurrently run `get_or_init`, more than one `f` can
    /// be called. However, all threads will return the same value, produced by
    /// some `f`.
    pub fn get_or_init<F>(&self, f: F) -> &'a T
    where
        F: FnOnce() -> &'a T,
    {
        enum Void {}
        match self.get_or_try_init(|| Ok::<&'a T, Void>(f())) {
            Ok(val) => val,
            Err(void) => match void {},
        }
    }

    /// Gets the contents of the cell, initializing it with `f` if
    /// the cell was empty. If the cell was empty and `f` failed, an
    /// error is returned.
    ///
    /// If several threads concurrently run `get_or_init`, more than one `f` can
    /// be called. However, all threads will return the same value, produced by
    /// some `f`.
    pub fn get_or_try_init<F, E>(&self, f: F) -> Result<&'a T, E>
    where
        F: FnOnce() -> Result<&'a T, E>,
    {
        let mut ptr = self.inner.load(Ordering::Acquire);

        if ptr.is_null() {
            // TODO replace with `cast_mut` when MSRV reaches 1.65.0 (also in `set`)
            ptr = f()? as *const T as *mut T;
            let exchange = self.inner.compare_exchange(
                ptr::null_mut(),
                ptr,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            if let Err(old) = exchange {
                ptr = old;
            }
        }

        Ok(unsafe { &*ptr })
    }

    /// ```compile_fail
    /// use once_cell::race::OnceRef;
    ///
    /// let mut l = OnceRef::new();
    ///
    /// {
    ///     let y = 2;
    ///     let mut r = OnceRef::new();
    ///     r.set(&y).unwrap();
    ///     core::mem::swap(&mut l, &mut r);
    /// }
    ///
    /// // l now contains a dangling reference to y
    /// eprintln!("uaf: {}", l.get().unwrap());
    /// ```
    fn _dummy() {}
}

#[cfg(feature = "alloc")]
pub use self::once_box::OnceBox;

#[cfg(feature = "alloc")]
mod once_box {
    use super::atomic::{AtomicPtr, Ordering};
    use core::{marker::PhantomData, ptr};

    use alloc::boxed::Box;

    /// A thread-safe cell which can be written to only once.
    pub struct OnceBox<T> {
        inner: AtomicPtr<T>,
        ghost: PhantomData<Option<Box<T>>>,
    }

    impl<T> core::fmt::Debug for OnceBox<T> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(f, "OnceBox({:?})", self.inner.load(Ordering::Relaxed))
        }
    }

    impl<T> Default for OnceBox<T> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<T> Drop for OnceBox<T> {
        fn drop(&mut self) {
            let ptr = *self.inner.get_mut();
            if !ptr.is_null() {
                drop(unsafe { Box::from_raw(ptr) })
            }
        }
    }

    impl<T> OnceBox<T> {
        /// Creates a new empty cell.
        pub const fn new() -> OnceBox<T> {
            OnceBox { inner: AtomicPtr::new(ptr::null_mut()), ghost: PhantomData }
        }

        /// Gets a reference to the underlying value.
        pub fn get(&self) -> Option<&T> {
            let ptr = self.inner.load(Ordering::Acquire);
            if ptr.is_null() {
                return None;
            }
            Some(unsafe { &*ptr })
        }

        /// Sets the contents of this cell to `value`.
        ///
        /// Returns `Ok(())` if the cell was empty and `Err(value)` if it was
        /// full.
        pub fn set(&self, value: Box<T>) -> Result<(), Box<T>> {
            let ptr = Box::into_raw(value);
            let exchange = self.inner.compare_exchange(
                ptr::null_mut(),
                ptr,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            if exchange.is_err() {
                let value = unsafe { Box::from_raw(ptr) };
                return Err(value);
            }
            Ok(())
        }

        /// Gets the contents of the cell, initializing it with `f` if the cell was
        /// empty.
        ///
        /// If several threads concurrently run `get_or_init`, more than one `f` can
        /// be called. However, all threads will return the same value, produced by
        /// some `f`.
        pub fn get_or_init<F>(&self, f: F) -> &T
        where
            F: FnOnce() -> Box<T>,
        {
            enum Void {}
            match self.get_or_try_init(|| Ok::<Box<T>, Void>(f())) {
                Ok(val) => val,
                Err(void) => match void {},
            }
        }

        /// Gets the contents of the cell, initializing it with `f` if
        /// the cell was empty. If the cell was empty and `f` failed, an
        /// error is returned.
        ///
        /// If several threads concurrently run `get_or_init`, more than one `f` can
        /// be called. However, all threads will return the same value, produced by
        /// some `f`.
        pub fn get_or_try_init<F, E>(&self, f: F) -> Result<&T, E>
        where
            F: FnOnce() -> Result<Box<T>, E>,
        {
            let mut ptr = self.inner.load(Ordering::Acquire);

            if ptr.is_null() {
                let val = f()?;
                ptr = Box::into_raw(val);
                let exchange = self.inner.compare_exchange(
                    ptr::null_mut(),
                    ptr,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                if let Err(old) = exchange {
                    drop(unsafe { Box::from_raw(ptr) });
                    ptr = old;
                }
            };
            Ok(unsafe { &*ptr })
        }
    }

    unsafe impl<T: Sync + Send> Sync for OnceBox<T> {}

    /// ```compile_fail
    /// struct S(*mut ());
    /// unsafe impl Sync for S {}
    ///
    /// fn share<T: Sync>(_: &T) {}
    /// share(&once_cell::race::OnceBox::<S>::new());
    /// ```
    fn _dummy() {}
}

pub use once_spin::OnceSpin;

mod once_spin {
    use super::atomic::{Ordering, AtomicUsize};
    use super::UnsafeCell;
    use core::mem::MaybeUninit;
    use core::hint::spin_loop;

    /// A thread-safe cell which can only be written to once. Note that the
    /// functions suffixed with `_spin` will spinloop in rare cases, and that
    /// other implementations are preferred if possible.
    #[derive(Debug)]
    pub struct OnceSpin<T> {
        /// The actual storage of the stored object
        data_holder: UnsafeCell<MaybeUninit<T>>,
        /// Tracks whether the OnceSpin has been initialized
        /// 0 -> no
        /// 1 -> write in progress (because writing to data_holder is not atomic)
        /// 2 -> init
        /// This value can only ever increase in increments of 1
        is_init: AtomicUsize,
        #[cfg(debug_assertions)]
        /// Helper counter to assert that the critical section is, in fact, only entered by one thread at a time
        critical_section_ctr: AtomicUsize
    }
    impl<T> Default for OnceSpin<T> {
        fn default() -> Self {
            Self::new()
        }
    }
    impl<T> OnceSpin<T> {
        /// Creates a new empty cell.
        #[inline]
        pub const fn new() -> Self {
            Self {
                data_holder: UnsafeCell::new(MaybeUninit::uninit()),
                is_init: AtomicUsize::new(0),
                #[cfg(debug_assertions)]
                critical_section_ctr: AtomicUsize::new(0)
            }
        }
        /// Creates a new initialized cell.
        #[inline]
        pub const fn with_value(value: T) -> Self {
            Self {
                data_holder: UnsafeCell::new(MaybeUninit::new(value)),
                is_init: AtomicUsize::new(2),
                #[cfg(debug_assertions)]
                critical_section_ctr: AtomicUsize::new(0)
            }
        }
        /// Gets a reference to the underlying value.
        ///
        /// SAFETY: callers must ensure that the OnceSpin is initialized.
        #[inline]
        pub unsafe fn get_unchecked(&self) -> &T {
            let mut_ptr = self.data_holder.get();
            (&*mut_ptr as &MaybeUninit<T>).assume_init_ref()
        }
        /// Gets a reference to the underlying value.
        pub fn get(&self) -> Option<&T> {
            let state_snapshot = self.is_init.load(Ordering::Acquire);
            if state_snapshot == 2 {
                #[cfg(debug_assertions)]
                assert_eq!(self.critical_section_ctr.load(Ordering::SeqCst), 0);
                // SAFETY: 2 -> value is init and nobody is trying to change it
                unsafe {
                    Some(self.get_unchecked())
                }
            } else {
                debug_assert!(state_snapshot <= 1);
                None
            }
        }
        /// Consumes the `OnceSpin`, returning the wrapped value.
        /// Returns `None` if the cell was empty.
        pub fn into_inner(mut self) -> Option<T> {
            let state_snapshot = self.is_init.load(Ordering::Acquire);
            if state_snapshot == 2 {
                #[cfg(debug_assertions)]
                assert_eq!(self.critical_section_ctr.load(Ordering::SeqCst), 0);
                // SAFETY: 2 -> value is init and nobody is trying to change it
                // This function requires `mut self` ownership and thus no
                // references to the underlying data can be live
                unsafe {
                    let cell = core::mem::replace(&mut self.data_holder, UnsafeCell::new(MaybeUninit::uninit()));
                    self.is_init.store(0, Ordering::Release);
                    Some(cell.into_inner().assume_init())
                }
            } else {
                debug_assert!(state_snapshot <= 1);
                None
            }
        }
        /// Forcibly sets the value of the cell and returns a mutable reference
        /// to the new value.
        ///
        /// SAFETY: The internal state must be set to the intermediate state before
        /// this is called. If the internal value was already set, then this
        /// function overwrites it without dropping it, potentially causing races.
        unsafe fn force_set(&self, value: T) -> &mut T {
            #[cfg(debug_assertions)]
            assert_eq!(self.critical_section_ctr.fetch_add(1, Ordering::SeqCst), 0);
            let value_ref: &mut T = unsafe {
                let mut_ptr = self.data_holder.get();
                (&mut *mut_ptr as &mut MaybeUninit<T>).write(value)
            };
            #[cfg(debug_assertions)]
            assert_eq!(self.critical_section_ctr.fetch_sub(1, Ordering::SeqCst), 1);
            value_ref
        }

        /// Sets the contents of this cell to `value`.
        ///
        /// Returns `Ok(())` if the cell was empty and `Err(value)` if it was full.
        pub fn set(&self, value: T) -> Result<(), T> {
            // Indicate that we are now trying to set the value
            // If someone else is also trying, back off and let them go through
            // On success we wish to release the new state, already knowing it
            // On failure we don't need an ordering, as the failure already forms
            // a happens-before relationship between their set and our check
            if self.is_init.compare_exchange(0, 1, Ordering::Release, Ordering::Relaxed).is_err() {
                return Err(value);
            }
            // SAFETY: state==1 -> nobody else is touching the UnsafeCell -> we can safely obtain &mut
            unsafe {
                self.force_set(value);
            }
            // Indicate that we have successfully written the value
            if self.is_init.swap(2, Ordering::AcqRel) != 1 {
                unreachable!("Concurrent modification to self.data_holder despite state signalling")
            }
            return Ok(())
        }

        /// Gets the contents of the cell, initializing it with `f` if the cell was
        /// empty.
        ///
        /// If several threads concurrently run `get_or_init`, more than one `f` can
        /// be called. However, all threads will return the same value, produced by
        /// some `f`.
        /// 
        /// Warning: In the rare case that one call to this function is writing its
        /// value to the cell while another execution of `f` has just finished, 
        /// then the latter call will spinloop until the cell is initialized before 
        /// returning a reference to the initialized value. The spinloop shouldn't 
        /// last long as it's only waiting for a one-time memory write, but if this
        /// is a problem for your application, consider alternatives such as the 
        /// cells backed by `critical_section`.
        pub fn get_or_init_spin<F>(&self, f: F) -> &T
        where
            F: FnOnce() -> T,
        {
            enum Void {}
            let fn_wrap = || {
                Ok::<T, Void> (f())
            };
            match self.get_or_try_init_spin(fn_wrap) {
                Ok(val) => val,
                Err(void) => match void {}
            }
        }
        /// Gets the contents of the cell, initializing it with `f` if
        /// the cell was empty. If the cell was empty and `f` failed, an
        /// error is returned.
        ///
        /// If several threads concurrently run `get_or_init`, more than one `f` can
        /// be called. However, all threads will return the same value, produced by
        /// some `f`.
        /// 
        /// Warning: In the rare case that one call to this function is writing its
        /// value to the cell while another execution of `f` has just finished, 
        /// then the latter call will spinloop until the cell is initialized before 
        /// returning a reference to the initialized value. The spinloop shouldn't 
        /// last long as it's only waiting for a one-time memory write, but if this
        /// is a problem for your application, consider alternatives such as the 
        /// cells backed by `critical_section`.
        pub fn get_or_try_init_spin<F, E>(&self, f: F) -> Result<&T, E>
        where
            F: FnOnce() -> Result<T, E>
        {
            let mut state_snapshot = self.is_init.load(Ordering::Acquire);

            if state_snapshot == 0 {
                let f_value = f()?;
                // Indicate that we are now trying to set the value
                // If someone else is also trying, break out and wait for the other write to go through
                // On success we wish to release the new state, without needing to acquire it again
                // On failure we need to acquire the actual state and wait for the other one setting the value to finish
                match self.is_init.compare_exchange(0, 1, Ordering::Release, Ordering::Acquire) {
                    Ok(_) => {
                        // SAFETY: state==1 -> nobody else is touching the UnsafeCell -> we can safely obtain &mut
                        let new_ref = unsafe {self.force_set(f_value)};
                        // Indicate that we have successfully written the value
                        if self.is_init.swap(2, Ordering::AcqRel) != 1 {
                            unreachable!("Concurrent modification to self.data_holder despite state signalling")
                        }
                        return Ok(new_ref as &T);
                    },
                    Err(new_state) => {
                        state_snapshot = new_state;
                        debug_assert!(state_snapshot==1 || state_snapshot==2);
                    }
                }
            }
            while state_snapshot == 1 {
                // 1 -> someone else is currently writing
                // Writes (should be) fast so we won't be spinning for long
                state_snapshot = self.is_init.load(Ordering::Acquire);
                spin_loop();
            }
            debug_assert_eq!(state_snapshot, 2);
            unsafe {
                Ok(self.get_unchecked())
            }
        }

        /// ```compile_fail
        /// # use once_cell::race::OnceSpin;
        /// #
        /// // Ensure that OnceSpin<T> is invariant over T lifetime subtypes
        /// let heap_object = std::vec::Vec::from([1,2,3,4]);
        /// let once_spin = OnceSpin::new();
        /// once_spin.set(&heap_object).unwrap();
        /// drop(heap_object);
        /// // The stored reference is no longer live because vec is dropped
        /// // The following line should fail to compile
        /// let _ref = once_spin.get();
        /// ```
        ///
        /// ``` compile_fail
        /// # use once_cell::race::OnceSpin;
        /// #
        /// // Ensure that OnceSpin<T> is invariant over T lifetime subtypes
        /// let heap_object = std::vec::Vec::from([1,2,3,4]);
        /// let once_spin = OnceSpin::new();
        /// once_spin.set(heap_object).unwrap();
        /// let obj_ref = once_spin.get().unwrap();
        /// let orig_vec = once_spin.into_inner().unwrap();
        /// // The stored reference is no longer live because vec is dropped
        /// // The following line should fail to compile
        /// println!("{}", obj_ref.len());
        /// ```
        fn _dummy() {}
    }

    impl<T> Drop for OnceSpin<T> {
        fn drop(&mut self) {
            let state = self.is_init.load(Ordering::Acquire);
            // &mut self -> nobody else can try to init -> value can't be 1
            // If we somehow do, then we leak the set value, which is safer than
            // incorrectly freeing it
            debug_assert_ne!(state, 1);
            if state == 2 {
                unsafe {
                    let mut_ptr = self.data_holder.get();
                    (&mut *mut_ptr as &mut MaybeUninit<T>).assume_init_drop();
                }
            }
        }
    }

    unsafe impl<T: Send+Sync> Sync for OnceSpin<T> {}

    #[cfg(test)]
    mod tests {
        
    }
}
