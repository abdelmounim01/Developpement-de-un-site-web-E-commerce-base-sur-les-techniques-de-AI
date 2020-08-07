import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { SearchFormvisiteurComponent } from './search-form.component';

describe('SearchFormComponent', () => {
  let component: SearchFormvisiteurComponent;
  let fixture: ComponentFixture<SearchFormvisiteurComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ SearchFormvisiteurComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(SearchFormvisiteurComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
