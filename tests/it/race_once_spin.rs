#[cfg(feature = "std")]
use std::sync::Barrier;
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    thread::scope,
};

use once_cell::race::OnceSpin;

#[test]
fn test_with_value() {
    let once_spin = OnceSpin::with_value(3);
    assert_eq!(*once_spin.get().unwrap(), 3);
}

#[test]
fn test_should_compile_static() {
    let heap_object = std::vec::Vec::from([1,2,3,4]);
    let once_storage = OnceSpin::new();
    once_storage.set(heap_object).unwrap();
    let _ref = once_storage.get();
    assert_eq!(_ref.unwrap(), &[1,2,3,4]);
    drop(once_storage);
}

#[test]
fn test_should_compile_nonstatic() {
    let heap_object = std::vec::Vec::from([1,2,3,4]);
    let once_storage = OnceSpin::new();
    once_storage.set(&heap_object).unwrap();
    let _ref = once_storage.get();
    drop(once_storage);
    drop(heap_object);
}

#[test]
fn test_init_only_once() {
    const THREAD_COUNT: usize = 20;

    let init_ctr = AtomicUsize::new(0);
    let barrier_obj = Barrier::new(THREAD_COUNT+1);
    let once_storage = OnceSpin::new();
    scope(|s| {
        // Start the threads...
        for _ in 0..THREAD_COUNT {
            s.spawn(|| {
                barrier_obj.wait();
                if once_storage.set(std::vec::Vec::from([std::string::String::from("abcd")])).is_ok() {
                    init_ctr.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        // ...and let them hammer the OnceStorage
        barrier_obj.wait();
    });
    // Ensure that writes to init_ctr are now visible
    std::sync::atomic::fence(Ordering::Acquire);
    // Check that object was only initialized once
    assert_eq!(init_ctr.load(Ordering::Acquire), 1);
    // Now read from the vec so that Miri can catch invalid accesses
    assert_eq!(once_storage.get().unwrap().len(), 1);
    assert_eq!(once_storage.get().unwrap()[0], "abcd");
}

#[test]
fn test_init_try_set_twice() {
    let once_spin = OnceSpin::new();

    assert!(once_spin.set(String::from("frist")).is_ok());
    assert_eq!(once_spin.get().unwrap(), "frist");

    assert!(once_spin.set(String::from("snecond")).is_err());
    assert_eq!(once_spin.get().unwrap(), "frist");
}

#[test]
fn test_drop_empty() {
    let x = OnceSpin::<String>::new();
    drop(x);
}

#[test]
fn test_reentrant() {
    let cell = OnceSpin::new();
    let res = cell.get_or_init_spin(|| {
        cell.get_or_init_spin(|| "hello".to_string());
        "world".to_string()
    });
    assert_eq!(res, "hello");
}
